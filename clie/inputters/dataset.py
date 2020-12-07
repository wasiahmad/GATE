import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from clie.inputters.vector import vectorize


class ACE05Dataset(Dataset):
    def __init__(self, examples, model, evaluation=False):
        self.model = model
        self.examples = examples
        self.evaluation = evaluation

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model,
                         iseval=self.evaluation)

    def lengths(self):
        return [len(ex.words) for ex in self.examples]


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l, np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
