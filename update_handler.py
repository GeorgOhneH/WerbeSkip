from deepnet import Network
import numpy as np
from deepnet.functions.costs import CrossEntropyCost
from helperfunctions.image_processing.retrieving_images import VideoCapture
from deepnet.layers import FullyConnectedLayer, BatchNorm, Dropout, ReLU, SoftMax, ConvolutionLayer, MaxPoolLayer, \
    Flatten
from deepnet.optimizers import Adam
import json
import websockets
import asyncio
from settings_secret import websocket_token
import warnings
import os


class WerbeSkip(object):
    def __init__(self):
        self.PATH_TO_NET = os.path.join(os.path.dirname(__file__),
                                        "helperfunctions/prosieben/networks/teleboy/teleboy_old.h5")
        self.ws = None
        self.loop = None
        self.network = self.init_network()
        self.docker = bool(os.environ.get("DJANGO_DEBUG", False))
        if self.docker:
            self.ip = "104.248.102.130:80"
        else:
            self.ip = "127.0.0.1:8000"
        # Prosieben: 354
        # SRF: 303
        self.cap = VideoCapture(channel=354, colour=False, rate_limit=1, convert_network=True, proxy=True, use_hash=False)

        self.filters = []
        self.result = []
        self.predictions = []

        self.filter_size = 25
        self.chain_size = 5

    def init_network(self):
        net = Network()

        net.input((1, 180, 320))

        net.add(
            ConvolutionLayer(n_filter=16, width_filter=12, height_filter=8, stride=4, zero_padding=0, padding_value=1))
        net.add(BatchNorm())
        net.add(ReLU())
        net.add(
            ConvolutionLayer(n_filter=64, width_filter=6, height_filter=4, stride=1, zero_padding=2, padding_value=1))
        net.add(BatchNorm())
        net.add(ReLU())
        net.add(MaxPoolLayer(width_filter=3, height_filter=3, stride=2))
        net.add(ConvolutionLayer(n_filter=128, width_filter=4, height_filter=4, stride=1))
        net.add(BatchNorm())
        net.add(ReLU())
        net.add(ConvolutionLayer(n_filter=128, width_filter=3, height_filter=3, stride=2))
        net.add(BatchNorm())
        net.add(ReLU())
        net.add(ConvolutionLayer(n_filter=256, width_filter=3, height_filter=3, stride=1))
        net.add(BatchNorm())
        net.add(Dropout(0.75))
        net.add(Flatten())
        net.add(FullyConnectedLayer(512))
        net.add(BatchNorm())
        net.add(ReLU())
        net.add(Dropout(0.5))
        net.add(FullyConnectedLayer(2))

        net.add(SoftMax())

        optimizer = Adam(learning_rate=0.001)
        net.regression(optimizer=optimizer, cost=CrossEntropyCost())
        net.load(self.PATH_TO_NET)
        return net

    async def init_db(self, websocket):
        message = {"command": "init", "channel": {"Prosieben": {"id": 354}}, "token": websocket_token}
        await websocket.send(json.dumps(message))

    async def producer_handler(self, websocket):
        while True:
            message = self.producer()
            await websocket.send(json.dumps(message))

    def producer(self):
        channel = self.get_prediction()
        message = {"command": "update", "room": 'main', "channel": channel, "token": websocket_token}
        return message

    def get_prediction(self):
        print("get image")
        img = next(self.cap)
        print("got image")

        prediction = self.network.feedforward(img)

        self.predictions.append(prediction[0, 1])
        snippet = self.predictions[-self.filter_size:]
        if np.any(np.array(snippet) > 0.9):  # checks if network is sure that it found a logo
            self.filters.append(1)
        else:
            self.filters.append(0)

        last_filter = self.filters[-1]
        if np.all(np.array(self.filters[-self.chain_size:]) == last_filter):  # checks if the last values are the same
            if last_filter == 1:
                if np.mean(self.predictions[-self.chain_size:]) > 0.9:
                    self.result.append(last_filter)
                else:
                    self.result.append(self.result[-1])
            else:
                self.result.append(last_filter)
        else:
            self.result.append(self.result[-1])

        self.clean_up()
        return {"Prosieben": {"ad": self.result[-1], "id": 354}}

    def clean_up(self):
        if len(self.predictions) > 2 * self.filter_size and len(self.filters) > 2 * self.chain_size:
            self.filters.pop(0)
            self.result.pop(0)
            self.predictions.pop(0)

    async def consumer_handler(self, websocket):
        while True:
            async for message in websocket:
                await self.consumer(message)
            asyncio.sleep(0)

    def consumer(self, message):
        print("message:", message)
        error = json.loads(message).get('error', None)
        if error:
            print('GOT ERROR FROM SOCKET:', error)

    async def handler(self, websocket):
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket))
        done, pending = await asyncio.wait(
            [consumer_task, producer_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        print("handler done")
        for task in pending:
            task.cancel()

    def run(self):
        async def hello():
            async with websockets.connect('ws://' + self.ip + '/chat/stream/') as websocket:
                print("connected")
                await self.init_db(websocket)
                await self.handler(websocket)
        print("starting")
        self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(hello())


if __name__ == "__main__":
        try:
            x = WerbeSkip()
            x.run()
        finally:
            x.cap.pipe.kill()  # not sure if pipe still runs after it shuts down or the programm exits
            x.cap.m3u8_update_thread.stop()  # stopping gracefully
            x.cap.get_images_thread.stop()  # stopping gracefully
            print("exit")
