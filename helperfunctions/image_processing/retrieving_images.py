import requests
import json
import cv2
import numpy as np
import math
import time


class VideoCapture(object):
    def __init__(self, channel: int, colour=True, rate_limit=None, convert_network=False):
        self.convert_network = convert_network
        self.rate_limit = rate_limit  # images per second
        self.colour = colour
        self.session = requests.session()
        self.setup_cookies()
        self.cap_url = self.get_cap_url(channel)
        self.cap = cv2.VideoCapture(self.cap_url)
        self.FRAME_RATE = self.cap.get(5)
        self.last_read = time.time()

    def setup_cookies(self):
        url = "https://www.teleboy.ch/api/anonymous/verify"
        response = self.session.get(url=url)
        data = response.json()
        token = data["data"]["_token"]

        data = {
            "_token": token,
            "age": 40,
            "gender": "male",
        }
        self.session.post(url=url, json=data)

    def get_cap_url(self, channel):
        url = "https://www.teleboy.ch/api/anonymous/live/{}".format(channel)
        response = self.session.get(url=url)
        j = json.loads(response.content)
        master_url = j["data"]["stream"]["url"]

        header = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:12.0) Gecko/20100101 Firefox/12.0'
        }

        response = self.session.get(master_url, verify=False, headers=header)
        data = response.content.decode("UTF-8")
        cap_url = data.split("\n")[2]
        return cap_url

    def __iter__(self):
        return self

    def __next__(self):
        if self.rate_limit:
            self.wait()

        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        if not self.colour:
            frame = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), axis=2)

        if self.convert_network:
            frame = np.expand_dims(frame.transpose((2, 0, 1)), axis=0) / 255

        return frame

    def grab(self, n):
        for _ in range(n):
            self.cap.grab()

    def wait(self):
        missed_frames = math.ceil(self.FRAME_RATE / self.rate_limit)
        self.grab(missed_frames)


if __name__ == "__main__":
    for frame in VideoCapture(channel=354):
        cv2.imshow("img", frame)
        cv2.waitKey(1)
