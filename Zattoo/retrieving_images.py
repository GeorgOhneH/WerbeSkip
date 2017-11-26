import urllib.request
import os
import warnings
from time import sleep


def save_img(url, path, name):
    fullfilename = os.path.join(path, name)
    if os.path.exists(fullfilename):
        warnings.warn("You are overwriting an existent file")
    urllib.request.urlretrieve(url, fullfilename)


if __name__ == "__main__":
    x = 0
    while True:
        save_img("https://thumb.zattic.com/prosieben/1024x576.jpg", "prosieben/images/unclassified", "F" + str(x) + ".jpg")
        print("image saved: %s" % x)
        x += 1
        sleep(15)
