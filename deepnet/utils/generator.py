from queue import Queue
from threading import Condition, Thread


class Generator(object):
    def __init__(self, epochs, mini_batch_size, n_workers):
        self.n_workers = n_workers
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.threads = []
        self.queue = Queue()
        self.cv = Condition()
        self.init()

    def init(self):
        indexes = [range(x, self.__len__(), self.n_workers) for x in range(self.n_workers)]
        self.threads = [Thread(target=self.produce_item, args=(index,)) for index in indexes]
        for thread in self.threads:
            thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        print(self.queue.qsize())
        with self.cv:
            while self.queue.empty():
                self.cv.wait()
            mini_batch = self.queue.get()
            self.queue.task_done()
            return mini_batch

    def __len__(self):
        raise NotImplemented

    def produce_item(self, indexes):
        for index in indexes:
            with self.cv:
                mini_batches = self.get_mini_batches(index)
                for mini_batch in mini_batches:
                    self.queue.put(mini_batch)

    def get_mini_batches(self, index) -> list:
        pass

