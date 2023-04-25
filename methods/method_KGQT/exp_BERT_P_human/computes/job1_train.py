from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer
from shared_modules.BERT_modules import BERTQPModule, BERTQPDataMoudle
import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
        model_name: str = typer.Option("bert-base-chinese"),
        batch_size: int = typer.Option(8),
        max_epoch: int = typer.Option(40),
        gpus: str = typer.Option([]),
        path_read_train_csv: Path = typer.Option("../stores/train_df.csv", help="读取train_df.csv的路径"),
        path_read_test_csv: Path = typer.Option("../stores/test_df.csv", help="读取train_df.csv的路径"),
        path_save_checkpoint_dir: Path = typer.Option("../outputs/"),
        project_name: str = typer.Option("exp_BERT_P"),
        check_val_every_n_epoch: int = typer.Option(1),
        path_cache_dir: Path = typer.Option(...),
        logging_dir: str = typer.Option(...),
        logging_name: str = typer.Option(...),
):
    # 读取训练数据
    train_df = pd.read_csv(path_read_train_csv,
                           names=["Pname", "Pid", "tsq", "nlq"])
    test_df = pd.read_csv(path_read_test_csv,
                          names=["Pname", "Pid", "tsq", "nlq"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=path_cache_dir)

    bert_qp_datamodule = BERTQPDataMoudle(
        train_df=train_df,
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=batch_size
    )

    bert_qp_module = BERTQPModule(
        model_name=model_name,
        num_labels=25,
        cache_dir=path_cache_dir
    )

    # 开始训练
    checkpoint_callback = ModelCheckpoint(
        dirpath=path_save_checkpoint_dir,
        filename="best-checkpoint",
        verbose=True,
        save_last=False,  # 因为训练的不是什么大模型，所以不需要
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # 监控的指标
        patience=5,  # 当指标在5个epoch内没有改善时停止训练
        verbose=True,  # 打印早停信息
        mode='min'  # 指标越小越好
    )

    logger = WandbLogger(
        project=project_name,
        name=logging_name,
        save_dir=logging_dir,

    )

    # 尝试训练
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        max_epochs=max_epoch,
        progress_bar_refresh_rate=30,
        gpus=gpus,
        accelerator="ddp",
        check_val_every_n_epoch=check_val_every_n_epoch
    )
    trainer.fit(bert_qp_module, bert_qp_datamodule)
    print(f"模型训练完成！训练脚本{__file__}")


if __name__ == "__main__":
    app()
