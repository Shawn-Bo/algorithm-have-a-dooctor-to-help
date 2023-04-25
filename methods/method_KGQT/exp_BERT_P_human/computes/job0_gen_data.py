import typer
import json
import math
import random
from pathlib import Path
import pandas as pd

app = typer.Typer()


@app.callback(invoke_without_command=True)  # 这个操作即可设置默认命令
def main(
        path_P_classes: str = typer.Option("../inputs/P_classes.json"),
        path_read_chatgpt_all: Path = typer.Option(...),
        path_read_human_all: Path = typer.Option(...),
        mode_train_dataset: str = typer.Option("human60p_only",
                                               help="生成训练集的模式，只有chatgpt_only和human60p_only两个选项。"),
        size_train_dataset: int = typer.Option(-1, help="生成训练集的大小，-1为全量数据集，其余正数为样本条数。"),
        path_save_train_df: Path = typer.Option("../stores/train_df.csv", help="保存train_df.csv的路径"),
        path_save_test_df: Path = typer.Option("../stores/test_df.csv", help="保存test_df.csv的路径")
):
    # 加载类别文件
    with Path(path_P_classes).open(encoding="utf-8", mode="r") as f:
        P_classes_dict = json.load(f)

    def generate_datasets_df(train_mode, size_train_dataset):
        def extract_questions_and_answers(question_path: Path):
            with question_path.open("r", encoding="utf-8") as f:
                data = []
                for line in f:
                    data_splits = line.replace("\n", "").split(",")
                    P = data_splits[0].split("【-】")[1]
                    data.append([P, P_classes_dict[P], data_splits[0], "".join(data_splits[1:])])
            return pd.DataFrame(data, columns=["P", "Pid", "tsq", "nlq"],
                                dtype=str)  # 这个名称是历史原因

        df_train, df_test = None, None

        df_chatgpt_all = extract_questions_and_answers(Path(path_read_chatgpt_all))
        df_human_all = extract_questions_and_answers(Path(path_read_human_all))

        # 按照输入大小取chatgpt数据集大小
        if size_train_dataset == -1:
            df_chatgpt_train = df_chatgpt_all
        else:
            df_chatgpt_train = df_chatgpt_all.sample(size_train_dataset)

        # 将人工数据集分割为训练集和测试集——按照S和P划分
        all_human_tsqs = df_human_all["tsq"].drop_duplicates().reset_index(drop=True).to_list()  # 长度为356
        test_human_tsqs = random.sample(all_human_tsqs, math.floor(0.4 * len(all_human_tsqs)))

        df_human_test = df_human_all[df_human_all["tsq"].isin(test_human_tsqs)]
        df_human_train = df_human_all.drop(df_human_test.index)

        # 提取训练集
        if train_mode == "chatgpt_only":
            df_train = df_chatgpt_train
            df_test = df_human_test
        elif train_mode == "human60p_only":
            df_train = df_human_train
            df_test = df_human_test
        else:
            print("train_mode 参数不正确！", train_mode)
            exit()
        return df_train, df_test

    train_df, test_df = generate_datasets_df(mode_train_dataset, size_train_dataset)

    # 保存文件
    train_df.to_csv(path_save_train_df, index=False, header=False)
    test_df.to_csv(path_save_test_df, index=False, header=False)
    print("数据集构建完成")


if __name__ == "__main__":
    app()