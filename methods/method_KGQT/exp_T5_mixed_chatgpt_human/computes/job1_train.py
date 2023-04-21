import pandas as pd
import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from shared_modules.T5_modules import T5Module, T5QTDatasetMoudle
from shared_modules.T5_pegasus_tokenizer import T5_pegasus_tokenizer

# 加载配置
with open("../inputs/config_job1_train.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

MODEL_NAME = config["model_name"]
CACHE_DIR_BASE = config["cache_dir_base"]
CHECKPOINT_PATH = config["checkpoint_path"]

BATCH_SIZE = config["batch_size"]
MAX_EPOCHS = config["max_epochs"]

LOGGING_DIR = config["logging_dir"]
LOGGING_NAME = config["logging_name"]

GPUS = config["gpus"]

# 读取训练数据
train_df = pd.read_csv(config["path_read_train_csv"], names=["masked_question", "origin_question"]).head(64)
test_df = pd.read_csv(config["path_read_test_csv"], names=["masked_question", "origin_question"])

# 加载数据模块
data_module = T5QTDatasetMoudle(train_df, test_df, T5_pegasus_tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

# 加载模型
model = T5Module()

# 开始训练
checkpoint_callback = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,
    filename="best-checkpoint",
    verbose=True,
    save_last=False,  # 因为训练的不是什么大模型，所以不需要
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',  # 监控的指标
    patience=3,  # 当指标在3个epoch内没有改善时停止训练
    verbose=True,  # 打印早停信息
    mode='min'  # 指标越小越好
)

logger = TensorBoardLogger(LOGGING_DIR, name=LOGGING_NAME)

trainer = pl.Trainer(
    logger=logger,
    callbacks = [checkpoint_callback, early_stop_callback],
    max_epochs=MAX_EPOCHS,
    gpus=GPUS,
    progress_bar_refresh_rate=30,
    accelerator="ddp"
)

trainer.fit(model, data_module)
print(f"模型训练完成！训练脚本{__file__}")

