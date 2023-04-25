import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertConfig
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer

class BERTQPModule(pl.LightningModule):
    def __init__(self,
                 model_name="bert-base-chinese",
                 num_labels=25,
                 cache_dir=None,
                 hidden_dropout_prob=0.5
                 ):
        super().__init__()
        bert_config = BertConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=hidden_dropout_prob,
            cache_dir=cache_dir
        )
        self.model = BertForSequenceClassification.from_pretrained(model_name, config=bert_config)
        self.model.train()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)


class BERTQPDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer,
            source_max_token_len: int = 64
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        nlq = data_row["nlq"]
        source_encoding = self.tokenizer(
            nlq,
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        Pids = torch.tensor(data_row["Pid"])
        return dict(
            nlq=nlq,
            Pid=Pids,
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=Pids.flatten()
        )


class BERTQPDataMoudle(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tokenizer,
            batch_size: int = 8,
            source_max_token_len: int = 64
    ):
        super().__init__()
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df

        self.train_dataset = BERTQPDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len
        )

        self.test_dataset = BERTQPDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 不再重新打乱数据，目的是便于分析训练过程。
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            num_workers=0
        )

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.test_dataset,
            batch_size=8,
            num_workers=0
        )
        return val_dataloader
