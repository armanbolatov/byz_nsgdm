"""Dataset loading for federated LM training."""

import os
import math
import torch
import urllib.request
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler


class CharDataset(Dataset):
    def __init__(self, data_dir, split="train", block_size=128, fraction=1.0):
        self.block_size = block_size
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, "shakespeare.txt")
        
        if not os.path.exists(data_file):
            print("Downloading Shakespeare dataset...")
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, data_file)
            print(f"Downloaded to {data_file}")
        
        with open(data_file, 'r') as f:
            text = f.read()
        
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = len(data)
        if split == "train":
            self.tokens = data[:int(0.9 * n)]
        else:
            self.tokens = data[int(0.9 * n):]
        
        if fraction < 1.0:
            n_tokens = int(len(self.tokens) * fraction)
            self.tokens = self.tokens[:n_tokens]
            print(f"Using {fraction*100:.0f}% of {split} data: {n_tokens} chars")
        
        self.num_samples = (len(self.tokens) - 1) // block_size
        print(f"Shakespeare {split}: {len(self.tokens)} chars, {self.num_samples} samples, vocab={self.vocab_size}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size + 1
        chunk = self.tokens[start:end]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class NonIIDSampler(Sampler):
    """Assigns contiguous chunks to each worker."""

    def __init__(self, dataset, num_replicas, rank, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        total_samples = len(dataset)
        self.samples_per_worker = total_samples // num_replicas
        self.start_idx = rank * self.samples_per_worker
        self.end_idx = self.start_idx + self.samples_per_worker
        
        if rank == num_replicas - 1:
            self.end_idx = total_samples
        
        self.num_samples = self.end_idx - self.start_idx

    def __iter__(self):
        indices = list(range(self.start_idx, self.end_idx))
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.rank * 1000)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class IIDSampler(Sampler):
    """Strided access across workers."""

    def __init__(self, dataset, num_replicas, rank, shuffle=True):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0
        self.num_samples = len(dataset) // num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # Distribute evenly (strided access = IID)
        indices = indices[self.rank::self.num_replicas][:self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
