from multiprocessing import Manager, Process
import numpy as np


class Generator(object):
    def __init__(self, epochs, mini_batch_size, n_workers=1):
        self.manager = Manager()
        self.n_workers = n_workers
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.threads = None
        self.progress = 0
        self.items = self.manager.list([])
        self.lock = self.manager.Lock()
        self.cv_produce = self.manager.Condition()
        self.cv_stop_produce = self.manager.Condition()

    def _init_generator(self):
        """Split the work into different threads"""
        del self.__dict__['manager']
        indexes = [range(x, self.__len__(), self.n_workers) for x in range(self.n_workers)]
        self.threads = [Process(target=self._produce_item, args=(index,)) for index in indexes]
        for thread in self.threads:
            thread.start()

    def _close(self):
        for thread in self.threads:
            thread.join()

    def __iter__(self):
        """Tells Python the class is the Iterator and init the class"""
        self._init_generator()
        return self

    def __next__(self):
        """Gets the items and makes the mini_batch"""
        with self.cv_produce:
            while self._items_len() < self.mini_batch_size:
                if not self._threads_alive():
                    self._close()
                    raise StopIteration
                self.cv_produce.wait(timeout=1)
            with self.lock:
                items = self.items[:self.mini_batch_size]
                del self.items[:self.mini_batch_size]
            self.progress += 1
        with self.cv_stop_produce:
            self.cv_stop_produce.notify()
        inputs = np.concatenate([item[0] for item in items], axis=0).astype('float32')
        labels = np.concatenate([item[1] for item in items], axis=0).astype('float32')
        return inputs, labels

    def _items_len(self):
        with self.lock:
            return len(self.items)

    def _threads_alive(self):
        for thread in self.threads:
            if thread.is_alive():
                return True
        return False

    def _produce_item(self, indexes):
        for index in indexes:
            with self.cv_produce:
                mini_batches = self.get_mini_batches(index)
                with self.lock:
                    self.items += mini_batches
                self.cv_produce.notify_all()
            with self.cv_stop_produce:
                while self._items_len() > self.mini_batch_size*2:
                    self.cv_stop_produce.wait()

    def __len__(self):
        raise NotImplemented

    def get_mini_batches(self, index) -> list:
        raise NotImplemented
