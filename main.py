from simple_loader import SimpleDataLoader
from torch.utils.data import DataLoader
import numpy as np
import time


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
