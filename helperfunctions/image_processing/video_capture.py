import requests
import json
import random
import numpy as np
import time
import os
import sys
import subprocess as sp
import threading
import queue
from urllib.parse import quote
import traceback
from settings_secret import proxy_username, proxy_password


class VideoCapture(object):
    """
    after https://github.com/reduzent/watchteleboy
    gets live images from teleboy
    """
    def __init__(self, channel: int, rate_limit=30, convert_network=False, colour=True, proxy=False, ffmpeg_log="error", use_hash=True):
        if proxy:
            parse_username = quote(proxy_username)
            parse_password = quote(proxy_password)
            self.proxies = {'https': 'https://' + parse_username + ':' + parse_password + '@proxy.mikrounix.com:3128'}
            self.test_proxy()
        else:
            self.proxies = {}

        self.pipe = None
        self.hash = "" if not use_hash else str(random.randint(1e10, 1e11))
        self.colour = colour
        self.ffmpeg_log = ffmpeg_log
        self.depth = 3 if colour else 1
        self.images = queue.Queue()
        self.m3u8_update_thread = None
        self.get_images_thread = None
        self.pipe_restart = None
        self.last_m3u8 = None
        self.convert_network = convert_network
        self.rate_limit = rate_limit  # frames per second
        self.session = requests.session()
        self.setup_cookies()
        self.cap_url = self.get_cap_url(channel)
        self.last_read = time.time()

        self.PATH_TO_CACHE = os.path.join(os.path.dirname(__file__), "m3u8_cache")
        self.M3U8_NAME = self.hash + "index.m3u8"
        self.ts_files = []
        self.init()

    def test_proxy(self):
        try:
            response = requests.get("https://www.google.com/", proxies=self.proxies)
            if response.status_code != 200:
                print("Response wan't successful, got status code: ", response.status_code, file=sys.stderr)
                self.exit_programm()
        except Exception:
            self.exit_programm()

    def exit_programm(self):
        traceback.print_exc()
        try:
            self.pipe.kill()  # not sure if pipe still runs after it shuts down or the programm exits
            self.m3u8_update_thread.stop()  # stopping gracefully
            self.get_images_thread.stop()  # stopping gracefully
            self.pipe_restart.stop()  # stopping gracefully
        except Exception:
            pass
        time.sleep(300)
        os._exit(1)

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
        for file in os.listdir(self.PATH_TO_CACHE):
            if ".ts" in file:
                os.remove(os.path.join(self.PATH_TO_CACHE, file))

        self.start_pipe()

        self.m3u8_update_thread = M3U8Updater(self)
        self.get_images_thread = GetImages(self)
        self.pipe_restart = PipeRestart(self)
        self.m3u8_update_thread.start()
        self.get_images_thread.start()
        self.pipe_restart.start()

    def start_pipe(self):
        self.update_m3u8_file()

        if self.colour:
            pix_fmt = 'bgr24'
        else:
            pix_fmt = 'gray'

        cmd_out = ['ffmpeg',
                   '-i', os.path.join(self.PATH_TO_CACHE, self.M3U8_NAME),
                   '-pix_fmt', pix_fmt,
                   '-c', 'copy',
                   '-vcodec', 'rawvideo',
                   '-probesize', '32',
                   '-loglevel', self.ffmpeg_log,
                   '-c:v', 'rawvideo',
                   '-f', 'image2pipe',
                   '-r', str(self.rate_limit),
                   '-']

        self.pipe = sp.Popen(cmd_out, bufsize=10 ** 8, stdout=sp.PIPE, stderr=sp.PIPE, cwd=self.PATH_TO_CACHE)

    def update_m3u8_file(self):
        text = requests.get(self.cap_url, proxies=self.proxies).text
        if text != self.last_m3u8:
            if len(self.ts_files) > 5:
                for file_name in self.ts_files[:-4]:
                    if os.path.exists(file_name):
                        try:
                            os.remove(file_name)
                        except FileNotFoundError:
                            pass
                    self.ts_files.remove(file_name)
            self.last_m3u8 = text
            for file_name in text.splitlines():
                path_to_ts_file = os.path.join(self.PATH_TO_CACHE, self.hash + file_name)
                if ".ts" in file_name and not os.path.exists(path_to_ts_file):
                    self.ts_files.append(path_to_ts_file)
                    file_url = self.cap_url[:-10] + file_name
                    data = requests.get(file_url, proxies=self.proxies).content
                    with open(path_to_ts_file, mode="wb") as f:
                        f.write(data)

            new_text = ""
            for i, line in enumerate(text.splitlines()):
                if "#EXT-X-MEDIA-SEQUENCE:" in line:
                    key, value = line.split(":")
                    line = key + ":" + self.hash + value
                elif ".ts" in line:
                    line = os.path.join(self.PATH_TO_CACHE, self.hash + line)
                new_text += line + "\n"
            with open(os.path.join(self.PATH_TO_CACHE, self.M3U8_NAME), mode="w") as f:
                f.write(new_text)

    def __iter__(self):
        return self

    def __next__(self):
        raw_image = self.images.get()

        frame = np.fromstring(raw_image, dtype='uint8').reshape((180, 320, self.depth)) / 255

        if self.convert_network:
            frame = np.expand_dims(frame.transpose((2, 0, 1)), axis=0)
        return frame


class StopThread(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._stop_event = threading.Event()

    def run(self):
        try:
            self.real_run()
        except Exception:
            self.cap.exit_programm()

    def real_run(self):
        raise NotImplemented

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class PipeRestart(StopThread):
    def __init__(self, cap):
        super().__init__(cap)

    def real_run(self):
        while True:
            err = self.cap.pipe.stderr.read(1).decode("UTF-8")
            while err[-1] != "\n":
                err += self.cap.pipe.stderr.read(1).decode("UTF-8")
            if err:
                print("Got error in pipe: {}".format(err), file=sys.stderr)
                print("restarting pipe...", file=sys.stderr)
                self.restart_pipe()
                print("restarted pipe", file=sys.stderr)
            time.sleep(300)
            if self.stopped():
                break

    def restart_pipe(self):
        self.cap.pipe.kill()
        print("pipe killed", file=sys.stderr)
        self.cap.start_pipe()


class M3U8Updater(StopThread):
    def __init__(self, cap):
        super().__init__(cap)

    def real_run(self):
        while True:
            time.sleep(0.5)
            self.cap.update_m3u8_file()
            if self.stopped():
                break


class GetImages(StopThread):
    def __init__(self, cap):
        super().__init__(cap)

    def real_run(self):
        while True:
            qsize = self.cap.images.qsize()
            if qsize > 5 * self.cap.rate_limit:
                for _ in range(qsize - 3*self.cap.rate_limit):
                    self.cap.images.get()
            raw_image = self.cap.pipe.stdout.read(180 * 320 * self.cap.depth)
            if raw_image:
                self.cap.images.put(raw_image)

            if self.stopped():
                break





