from torch.utils.data import Dataset
import numpy as np
import torch 

class LMDataset(Dataset):
    def __init__(self, tokens_path, context_length, max_tokens=None):
        self.tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r')
        self.context_length = context_length
        if max_tokens is None:
            self.num_sequences = (len(self.tokens) - 1) // context_length
        else:
            # limit to max_tokens
            self.num_sequences = min((len(self.tokens) - 1) // context_length,
                                     max_tokens // context_length)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start = idx * self.context_length

        input_tokens = self.tokens[start:start+self.context_length]
        label_tokens = self.tokens[start+1:start+self.context_length+1]

        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(label_tokens, dtype=torch.long)

        return input_tensor, target_tensor