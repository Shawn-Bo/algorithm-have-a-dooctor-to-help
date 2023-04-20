
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration

with next(Path(__file__).parent.glob("config_shared.yaml")).open(mode="r") as stream:
    config = yaml.safe_load(stream)


class T5Module(pl.LightningModule):
    def __init__(self):
        super().__init__()
        MODEL_NAME = config["model_name"]
        MODEL_BASE = config["model_base"]
        self.model = MT5ForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            cache_dir=f"{MODEL_BASE}/{MODEL_NAME}",
            return_dict=True)
        self.model.resize_token_embeddings(50002)

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
    
    
class T5QTDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer,
            source_max_token_len: int = 64,
            target_max_token_len: int = 64
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        source_encoding = self.tokenizer(
            data_row[1],
            max_length=self.source_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            data_row[0],
            max_length=self.target_max_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
        labels = target_encoding["input_ids"]
        return dict(
            masked_question=data_row[1],
            origin_question=data_row[0],
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )
        
        
class T5QTDatasetMoudle(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer,
        batch_size: int = 8,
        source_max_token_len: int = 64,
        target_max_token_len: int = 64
    ):
        super().__init__()
        self.test_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.test_df = test_df

    def setup(self):
        self.train_dataset = T5QTDataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = T5QTDataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
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
        return DataLoader(
            self.test_dataset,
            batch_size=8,
            num_workers=0
        )
