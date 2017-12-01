import pygame
from pygame.locals import *
import os
import shutil
from sys import exit
from PIL import Image


class Window(object):
    """
    Makes a Pygamewindow where all images from a directory will be displayed
    and will be moved to the correct directorys as soon the window is closed.
    It has 2 inputs:
    1. The path to one directory were all the images are located
    2. A list with all paths to the directorys were the image should be moved to.

    You can switch between images with the arrowkeys left and right
    To move an image to a other directory press the correct nummber where
    1 equals to the first item in the list were the paths to classes are
    2 equals to the second one and so on
    """
    def __init__(self, path_to_images, paths_to_classes):
        pygame.init()
        self.class_keys = [K_1, K_2, K_3, K_4, K_5, K_6, K_7, K_8, K_9, K_0]
        self.classes_paths = paths_to_classes
        self.image_paths = [os.path.join(path_to_images, image_path) for image_path in os.listdir(path_to_images)]
        self.images = [{"path": image_path, "class_path": None} for image_path in self.image_paths]
        self.width, self.height = Image.open(self.image_paths[0]).size
        self.screen = pygame.display.set_mode((self.width, self.height))

    def start(self):
        index = 0
        lock_next = False
        while True:
            for event in pygame.event.get():
                # removes lock
                if event.type == KEYUP:
                    lock_next = False
                # closes windows and moves the images
                if event.type == KEYDOWN and event.key == K_ESCAPE or event.type == QUIT:
                    self.move_imgs()
                    pygame.quit()
                    exit()
            pressed_keys = pygame.key.get_pressed()
            # Alt + f4 closes window without saving
            if pressed_keys[K_F4] and pressed_keys[K_LALT]:
                pygame.quit()
                exit()
            # assign image and move to the next one
            for index_key, key in enumerate(self.class_keys):
                if pressed_keys[key] and not lock_next:
                    self.images[index]["class_path"] = self.classes_paths[index_key]
                    lock_next = True
                    index += 1
                    break
            # step through the images
            if pressed_keys[K_LEFT] and not lock_next:
                lock_next = True
                index -= 1
            elif pressed_keys[K_RIGHT] and not lock_next:
                lock_next = True
                index += 1
            # prevents overindexing
            index %= len(self.images)
            # loads image
            image = pygame.image.load(self.images[index]["path"])
            # displays image
            self.screen.blit(image, (0, 0))

            # makes the font and displays it
            font = pygame.font.SysFont("arial", 50)
            text = font.render(str(self.images[index]["class_path"]), True, (200, 200, 0))
            name = font.render(self.images[index]["path"].split("d")[-1], True, (200, 200, 0))
            self.screen.blit(text, (0, 0))
            self.screen.blit(name, (0, self.height - 50))

            # updates screen
            pygame.display.update()

    def move_imgs(self):
        for image in self.images:
            if image["class_path"] is not None:
                shutil.move(image["path"], image["class_path"])


if __name__ == "__main__":
    path_to_images = "prosieben/images/unclassified"
    paths_to_classes = [
        "prosieben/images/classified/logo_boarder_left_right",
        "prosieben/images/classified/logo_boarder_up_down",
        "prosieben/images/classified/logo",
        "prosieben/images/classified/no_logo",
        "prosieben/images/classified/special",
    ]
    window = Window(path_to_images, paths_to_classes)
    window.start()
