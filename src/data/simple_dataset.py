from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):

    """DataFrame을 Dataset으로 감싸는 최소한의 래퍼"""
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)  # 인덱스 리셋
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()  # 행을 딕셔너리로 반환