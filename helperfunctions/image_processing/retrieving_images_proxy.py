import requests
import json
import cv2
import numpy as np
import time
import os
import subprocess as sp
from threading import Thread
import queue


class VideoCapture(object):
    def __init__(self, channel: int, rate_limit=30, convert_network=False):
        self.proxies = {}

        self.pipe = None
        self.images = queue.Queue()
        self.m3u8_update_thread = None
        self.get_images_t = None
        self.last_m3u8 = None
        self.convert_network = convert_network
        self.rate_limit = rate_limit  # frames per second
        self.session = requests.session()
        self.setup_cookies()
        self.cap_url = self.get_cap_url(channel)
        self.last_read = time.time()

        self.PATH_TO_CACHE = os.path.join(os.path.dirname(__file__), "m3u8_cache")
        self.PATH_TO_TS_FILES = os.path.dirname(os.path.dirname(__file__))  # ffmpeg is a bit weird with the paths
        self.M3U8_NAME = "index.m3u8"
        self.ts_files = []
        self.init()

    def setup_cookies(self):
        url = "https://www.teleboy.ch/api/anonymous/verify"
        response = self.session.get(url=url, proxies=self.proxies)
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
        response = self.session.get(url=url, proxies=self.proxies)
        j = json.loads(response.content)
        master_url = j["data"]["stream"]["url"]

        header = {
            'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:12.0) Gecko/20100101 Firefox/12.0'
        }

        response = self.session.get(master_url, verify=False, headers=header, proxies=self.proxies)
        data = response.content.decode("UTF-8")
        cap_url = data.split("\n")[2]
        return cap_url

    def init(self):
        for file in os.listdir(self.PATH_TO_TS_FILES):
            if ".ts" in file:
                os.remove(os.path.join(self.PATH_TO_TS_FILES, file))

        self.update_m3u8_file()

        cmd_out = ['ffmpeg ',
                   '-i', os.path.join(self.PATH_TO_CACHE, self.M3U8_NAME),  # Indicated input comes from pipe
                   '-pix_fmt', 'gray',
                   '-c', 'copy',
                   '-vcodec', 'rawvideo',
                   '-probesize', '32',
                   '-c:v', 'rawvideo',
                   '-f', 'image2pipe',
                   '-r', str(self.rate_limit),  # framerate
                   '-']

        self.pipe = sp.Popen(cmd_out, bufsize=10 ** 8, stdout=sp.PIPE)
        self.m3u8_update_thread = Thread(target=self.m3u8_thread)
        self.get_images_t = Thread(target=self.get_images_thread)
        self.m3u8_update_thread.start()
        self.get_images_t.start()

    def update_m3u8_file(self):
        text = requests.get(self.cap_url, proxies=self.proxies).text
        if text != self.last_m3u8:
            if len(self.ts_files) > 5:
                for file_name in self.ts_files[:-4]:
                    os.remove(file_name)
                    self.ts_files.remove(file_name)
            self.last_m3u8 = text
            for file_name in text.splitlines():
                path_to_ts_file = os.path.join(self.PATH_TO_TS_FILES, file_name)
                if ".ts" in file_name and not os.path.exists(path_to_ts_file):
                    self.ts_files.append(path_to_ts_file)
                    file_url = self.cap_url[:-10] + file_name
                    data = requests.get(file_url, proxies=self.proxies).content
                    with open(path_to_ts_file, mode="wb") as f:
                        f.write(data)
            with open(os.path.join(self.PATH_TO_CACHE, self.M3U8_NAME), mode="w") as f:
                f.write(text)

    def m3u8_thread(self):
        while True:
            time.sleep(0.5)
            self.update_m3u8_file()

    def get_images_thread(self):
        while True:
            qsize = self.images.qsize()
            if qsize > 5 * self.rate_limit:
                for _ in range(qsize - 3*self.rate_limit):
                    self.images.get()

            raw_image = self.pipe.stdout.read(180 * 320 * 1)
            frame = np.fromstring(raw_image, dtype='uint8').reshape((180, 320, 1)) / 255

            if self.convert_network:
                frame = np.expand_dims(frame.transpose((2, 0, 1)), axis=0)

            self.images.put(frame)

    def __iter__(self):
        return self

    def __next__(self):
        frame = self.images.get()
        return frame


if __name__ == "__main__":
    for frame in VideoCapture(channel=354, rate_limit=30):
        cv2.imshow("img", frame)
        cv2.waitKey(1)
