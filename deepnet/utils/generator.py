from threading import Condition, Thread, Lock
import numpy as np


class Generator(object):
    def __init__(self, epochs, mini_batch_size, n_workers=2):
        self.n_workers = n_workers
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.threads = None
        self.items = []
        self.lock = None
        self.cv_produce = None
        self.cv_stop_produce = None

    def init_generator(self):
        """Split the work into different threads"""
        self.items = []
        self.lock = Lock()
        self.cv_produce = Condition()
        self.cv_stop_produce = Condition()
        indexes = [range(x, self.__len__(), self.n_workers) for x in range(self.n_workers)]
        self.threads = [Thread(target=self.produce_item, args=(index,)) for index in indexes]
        for thread in self.threads:
            thread.start()

    def close(self):
        pass

    def __iter__(self):
        """Tells Python the class is the Iterator and init the class"""
        self.init_generator()
        return self

    def __next__(self):
        """Gets the items and makes the mini_batch"""
        with self.cv_produce:
            while self.items_len() < self.mini_batch_size:
                if not self.threads_alive():
                    self.close()
                    raise StopIteration
                self.cv_produce.wait()
            with self.lock:
                items = self.items[:self.mini_batch_size]
                del self.items[:self.mini_batch_size]
        with self.cv_stop_produce:
            self.cv_stop_produce.notify_all()
        inputs = np.concatenate([item[0] for item in items], axis=1)
        labels = np.concatenate([item[1] for item in items], axis=1)
        return inputs, labels

    def items_len(self):
        with self.lock:
            return len(self.items)

    def threads_alive(self):
        for thread in self.threads:
            if thread.is_alive():
                return True
        return False

    def produce_item(self, indexes):
        for index in indexes:
            with self.cv_produce:
                mini_batches = self.get_mini_batches(index)
                with self.lock:
                    self.items += mini_batches
                self.cv_produce.notify_all()
            with self.cv_stop_produce:
                while self.items_len() > self.mini_batch_size*10:
                    self.cv_stop_produce.wait()

    def __len__(self):
        raise NotImplemented

    def get_mini_batches(self, index) -> list:
        raise NotImplemented

