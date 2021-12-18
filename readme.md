# 一个简易的DataLoader

​	这是一个简易的dataloader，把pytorch的dataloader核心部分用python自带的函数重写了一遍。加载速度比原先的pytorch的dataloader更快，如果不是特别复杂的任务如ddp这些，基本能够代替pytorch的dataloader，使用方法上也没有差别。



# 使用方法

​	支持pytorch dataloader 中常见的参数接口。运行的示例代码如下

```python
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
```

结果如下：

`pytorch DataLoader use time 42507.71, total data=55`
`CDataLoader use time 865.67, total data=55`

