import multiprocessing
import numpy as np
import time
import random
from torch.utils.data import DataLoader
from itertools import cycle
from utils import BatchSampler, default_collate, put_data

class FakeDataset:

    def __init__(self):
        # 制造一些假数据
        self.imgs = [np.random.random((28, 28)) for i in range(45)]

    def __getitem__(self, item):
        # 模拟图像处理时间
        for i in range(1000000):
            pass
        return self.imgs[item]

    def __len__(self):
        return len(self.imgs)


class SimpleDataLoader:

    def __init__(self, dataset, batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=None):
        num_workers = max(1, num_workers)
        self.dataset = dataset
        self.drop_last = drop_last
        self.index_queue = []
        sampler = range(len(self.dataset)) if not shuffle else random.sample(range(len(dataset)), len(dataset))
        self.batch_index_sampler = BatchSampler(sampler, batch_size, drop_last=drop_last)
        self.num_workers = num_workers
        self.worker_queue_index = cycle(range(num_workers))
        self.image_queue = multiprocessing.Queue()
        self.num_dataset = len(dataset)
        self.batch_size = batch_size
        if collate_fn is None:
            collate_fn = default_collate

        for i in range(num_workers):
            index_queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=put_data, args=(dataset, index_queue, self.image_queue, collate_fn))
            p.daemon = True
            p.start()
            self.index_queue.append(index_queue)
            pass

    def __iter__(self):
        self.reset()
        last_batch = 1 if not self.drop_last and self.num_dataset % self.batch_size else 0
        iter_num = self.num_dataset // self.batch_size + last_batch
        for i in range(iter_num):
            data = self.image_queue.get()
            yield data
            self.try_put_index()


    def reset(self):
        self.batch_index_iter = iter(self.batch_index_sampler)
        for _ in range(self.num_workers):
           self.try_put_index()

    def try_put_index(self):
        try:
            batch_index = next(self.batch_index_iter)
            worker_index = next(self.worker_queue_index)
            self.index_queue[worker_index].put(batch_index)
        except StopIteration:
            return


if __name__ == "__main__":
    dataset = FakeDataset()
    epoch = 5

    d = []
    loader = DataLoader(dataset, batch_size=4, num_workers=8, drop_last=True)
    start = time.time()
    for _ in range(epoch):
        for data in loader:
            d.append(data)
    end = time.time()
    print("pytorch DataLoader use time %.2f, total data=%d"%((end-start) * 1000, len(d)))

    loader = SimpleDataLoader(dataset, batch_size=4, num_workers=8, shuffle=True, drop_last=True)
    d = []
    start = time.time()
    for _ in range(epoch):
        for data in loader:
            d.append(data)

        pass
    end = time.time()
    print("CDataLoader use time %.2f, total data=%d" % ((end - start) * 1000, len(d)))
