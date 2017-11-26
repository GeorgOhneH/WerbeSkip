import pygame
from pygame.locals import *
import os
import shutil
from sys import exit
from PIL import Image


class Window(object):
    def __init__(self, path_to_images, paths_to_classes):
        pygame.init()
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
                if event.type == KEYUP:
                    lock_next = False
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        self.move_imgs()
                        exit()
            pressed_keys = pygame.key.get_pressed()
            if pressed_keys[K_1] and not lock_next:
                self.images[index]["class_path"] = self.classes_paths[0]
                lock_next = True
                index += 1
            elif pressed_keys[K_2] and not lock_next:
                self.images[index]["class_path"] = self.classes_paths[1]
                lock_next = True
                index += 1
            elif pressed_keys[K_3] and not lock_next:
                self.images[index]["class_path"] = self.classes_paths[2]
                lock_next = True
                index += 1
            elif pressed_keys[K_4] and not lock_next:
                self.images[index]["class_path"] = self.classes_paths[3]
                lock_next = True
                index += 1
            elif pressed_keys[K_5] and not lock_next:
                self.images[index]["class_path"] = self.classes_paths[4]
                lock_next = True
                index += 1
            elif pressed_keys[K_LEFT] and not lock_next:
                lock_next = True
                index -= 1
            elif pressed_keys[K_RIGHT] and not lock_next:
                lock_next = True
                index += 1
            index %= len(self.images)
            image = pygame.image.load(self.images[index]["path"])
            self.screen.blit(image, (0, 0))
            font = pygame.font.SysFont("arial", 50)
            text = font.render(str(self.images[index]["class_path"]), True, (200, 200, 0))
            name = font.render(self.images[index]["path"].split("d")[-1], True, (200, 200, 0))
            self.screen.blit(text, (0, 0))
            self.screen.blit(name, (0, self.height - 50))
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
