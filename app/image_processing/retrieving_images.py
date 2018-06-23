import urllib.request
import os
import warnings
import time


def save_img(path):
    creation_date = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
    fullfilename = os.path.join(path, creation_date + ".jpg")
    if os.path.exists(fullfilename):
        warnings.warn("You are overwriting an existent file")
    urllib.request.urlretrieve("https://thumb.zattic.com/prosieben/1024x576.jpg", fullfilename)


if __name__ == "__main__":
    x = 0
    while True:
        save_img("app/prosieben/images/unclassified")
        print("image saved: %s" % x)
        x += 1
        time.sleep(15)
