from torch.utils.data import Dataset


class ValleDataset(Dataset):
    def __init__(self, tokens_list, codes_list):
        self.tokens_list = tokens_list
        self.codes_list = codes_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return self.tokens_list[idx], self.codes_list[idx]
