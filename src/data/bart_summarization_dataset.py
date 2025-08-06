import pandas as pd
from torch.utils.data import Dataset

from src import config
from src.utils.helper import preprocess

class BartSummarizationDataset(Dataset):
    def __init__(self, cfg, file_path, tokenizer, is_train):

        self.bos_token = cfg.tokenizer.bos_token
        self.eos_token = cfg.tokenizer.eos_token
        
        self.df = make_set_as_df(file_path, is_train)
        self.encoder_input_train , self.decoder_input_train, self.decoder_output_train = self.make_input(is_train)
        self.encoder_input , self.decoder_input, self.labels = self.tokenizing(cfg, tokenizer, is_train)
        self.len = len(self.encoder_input_train)
        self.is_train = is_train
        
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]

        if(self.is_train == True):
            item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
            item2['decoder_input_ids'] = item2['input_ids']
            item2['decoder_attention_mask'] = item2['attention_mask']
            item2.pop('input_ids')
            item2.pop('attention_mask')
            item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
            item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        else:
            pass
            #item['ID'] = self.test_id[idx]   
        return item

    def __len__(self):
        return self.len
    
    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, is_train = True):

        dataset = self.df
        
        if is_train:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # Ground truth를 디코더의 input으로 사용하여 학습합니다.
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()    
        else:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input), None
        
    # tokenizing
    def tokenizing(self, cfg, tokenizer, is_train = True):

        tokenized_encoder_inputs = tokenizer(self.encoder_input_train, return_tensors="pt", padding=True,
                                add_special_tokens=True, truncation=True, max_length=cfg.tokenizer.encoder_max_len1, return_token_type_ids=False)
        tokenized_decoder_inputs = tokenizer(self.decoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=cfg.tokenizer.encoder_max_len2, return_token_type_ids=False)
        tokenized_decoder_ouputs = tokenizer(self.decoder_output_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=cfg.tokenizer.encoder_max_len2, return_token_type_ids=False) if is_train else None
         
        return tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs


# 실험에 필요한 컬럼을 가져옵니다.
def make_set_as_df(file_path, is_train = True):
    if is_train:
        df = pd.read_csv(file_path)
        train_df = df[['fname','dialogue','summary']].copy()
        train_df['dialogue'] = train_df['dialogue'].apply(preprocess)
        return train_df
    else:
        df = pd.read_csv(file_path)
        test_df = df[['fname','dialogue']].copy()
        test_df['dialogue'] = test_df['dialogue'].apply(preprocess)
        return test_df
    


def get_datasets(cfg, tokenizer):
    # Dataset 정의
    train_dataset = BartSummarizationDataset(cfg, config.RAW_DATA_DIR / cfg.custom.train_file_name, tokenizer, is_train=True)

    val_dataset = BartSummarizationDataset(cfg, config.RAW_DATA_DIR / cfg.custom.val_file_name, tokenizer, is_train=True)

    test_dataset = BartSummarizationDataset(cfg, config.RAW_DATA_DIR / cfg.custom.test_file_name, tokenizer, is_train=False)

    return train_dataset, val_dataset, test_dataset 
