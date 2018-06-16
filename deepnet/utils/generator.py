from queue import Queue
from threading import Condition, Thread
import numpy as np


class Generator(object):
    def __init__(self, epochs, mini_batch_size, n_workers=2):
        self.n_workers = n_workers
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.threads = []
        self.queue = Queue()
        self.cv = Condition()
        self.init_generator()

    def init_generator(self):
        """Split the work into different threads"""
        indexes = [range(x, self.__len__(), self.n_workers) for x in range(self.n_workers)]
        self.threads = [Thread(target=self.produce_item, args=(index,)) for index in indexes]
        for thread in self.threads:
            thread.start()

    def __iter__(self):
        """Tells Python the class is the Iterator"""
        return self

    def __next__(self):
        """Gets the items and makes the mini_batch"""
        print(self.queue.qsize())
        inputs, labels = [], []
        for _ in range(self.mini_batch_size):
            with self.cv:
                while self.queue.empty():
                    if not self.threads_alive():
                        return StopIteration
                    self.cv.wait()
                item = self.queue.get()
                inputs.append(item[0])
                labels.append(item[1])
                self.queue.task_done()
        inputs = np.concatenate(inputs, axis=1)
        labels = np.concatenate(labels, axis=1)
        return inputs, labels

    def threads_alive(self):
        for thread in self.threads:
            if thread.is_alive():
                return True
        return False

    def produce_item(self, indexes):
        for index in indexes:
            with self.cv:
                mini_batches = self.get_mini_batches(index)
                for mini_batch in mini_batches:
                    self.queue.put(mini_batch)
                self.cv.notify_all()

    def __len__(self):
        raise NotImplemented

    def get_mini_batches(self, index) -> list:
        raise NotImplemented

