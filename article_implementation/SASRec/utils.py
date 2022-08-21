from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm

class TransactionDataset(Dataset):
    def __init__(self, df_train, min_sample_len = 10):
        self.df_train = df_train
        self.df_train['timestamp'] = pd.to_datetime(self.df_train['timestamp'])
        self.df_train = self.df_train.sort_values(by=['timestamp'])
        users = set(self.df_train['userId'])
        self.transactions = []
        for user_id in tqdm(users):
            users_transactions = np.array(self.df_train[self.df_train['userId'] == user_id]['movieId'].values)
            if len(users_transactions) < min_sample_len:
                continue
            for i in range(len(users_transactions) - min_sample_len + 1):
                end_pos = i + min_sample_len
                self.transactions.append(users_transactions[i:end_pos])
        
    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, index):
        return torch.LongTensor(self.transactions[index]).to(device)
