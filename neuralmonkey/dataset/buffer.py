import random


class TrainingBuffer:
    """ This class holds a set of on-the-fly training instances """

    def __init__(self, max_size: int, log_file: str):
        self.max_size = max_size
        self.deque = []
        self.log = open(log_file, 'a')

    def log_new_entry(self, entry):
        self.log.write(str(entry))
        self.log.flush()

    def shuffle(self):
        random.shuffle(self.deque)

    def add_batch(self, new_items):
        for new_item in new_items:
            self.log_new_entry(new_item)
        # make space for new items
        while len(self.deque) > self.max_size-len(new_items):
            self.deque.pop()
        self.deque.extend(new_items)
        self.shuffle()
        assert len(self.deque) <= self.max_size

    def add_single(self, new_item):
        while len(self.deque) > self.max_size-1:
            self.deque.pop()
        self.log_new_entry(new_item)
        self.deque.append(new_item)
        self.shuffle()
        assert len(self.deque) <= self.max_size

    def batch_dataset(self, size):
        """
        Get batch of specified size
        :param size:
        :return: list of WeightedTrainInstances
        """
        assert len(self.deque) >= size
        batch = []
        while len(batch) < size and len(self.deque) > 0:
            batch.append(self.deque.pop())
        return batch

    def is_empty(self):
        return len(self.deque) == 0

    def __len__(self):
        return len(self.deque)


class WeightedTrainInstance:
    def __init__(self, src, trg, reward, logprob):
        self.src = src
        self.trg = trg
        self.reward = reward
        self.logprob = logprob

    def __str__(self):
        return "{}\t{}\t{}\t{}\n".format(" ".join(self.src), " ".join(self.trg),
                                         self.reward, self.logprob)

    def __len__(self):
        return len(self.src)+len(self.trg)



