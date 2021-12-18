import torch


def put_data(dataset, index_queue, image_queue, collate_fn):
    while True:
        batch_index = index_queue.get()
        data = [dataset[i] for i in batch_index]
        data = collate_fn(data)
        image_queue.put(data)
        del data, batch_index


def default_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy':
        return default_collate([torch.as_tensor(b) for b in batch])
    else:
        raise NotImplementedError

class BatchSampler:

    def __init__(self, sampler, batch_size, drop_last):
        super(BatchSampler, self).__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):

        batch = []
        for idx in self.sampler:

            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
