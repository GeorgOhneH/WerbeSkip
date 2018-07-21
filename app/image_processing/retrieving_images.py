import requests
import json
import cv2
import numpy as np


class VideoCapture(object):
    def __init__(self, channel: int):
        self.session = requests.session()
        self.setup_cookies()
        self.cap_url = self.get_cap_url(channel)
        self.cap = cv2.VideoCapture(self.cap_url)

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
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame


if __name__ == "__main__":
    smallest_m = 100000000000
    for frame in VideoCapture(channel=303):
        m = np.mean(frame)
        if m < smallest_m:
            cv2.imwrite("black_img.png", frame)
            smallest_m = m
