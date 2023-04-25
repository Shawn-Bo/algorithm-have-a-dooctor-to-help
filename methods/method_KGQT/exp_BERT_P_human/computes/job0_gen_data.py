"""
    为模型训练生成数据，使用随机种子。
    输入：ChatGPT和Human数据集
    输出：分割好的数据保存在shared中
"""

import math
import random
from pathlib import Path

import pandas as pd
import yaml

with open("../inputs/config_job0_gen_data.yaml", 'r') as stream:
    config = yaml.safe_load(stream)


def generate_datasets_df(train_mode, size_train_dataset):
    def extract_questions_and_answers(question_path: Path):
        with question_path.open("r", encoding="utf-8") as f:
            data = []
            for line in f:
                data_splits = line.replace("\n", "").split(",")
                data.append([data_splits[0], "".join(data_splits[1:])])
        return pd.DataFrame(data, columns=["masked_question", "origin_question"], dtype=str)  # 这个名称是历史原因

    df_train, df_test = None, None

    df_chatgpt_all = extract_questions_and_answers(Path(config["path_read_chatgpt_all"]))
    df_human_all = extract_questions_and_answers(Path(config["path_read_human_all"]))

    # 按照输入大小取chatgpt数据集大小
    if size_train_dataset == "all":
        df_chatgpt_train = df_chatgpt_all
    else:
        df_chatgpt_train = df_chatgpt_all.sample(size_train_dataset)

    # 将人工数据集分割为训练集和测试集——按照S和P划分    
    all_human_tsqs = df_human_all["masked_question"].drop_duplicates().reset_index(drop=True).to_list()  # 长度为356
    test_human_tsqs = random.sample(all_human_tsqs, math.floor(0.4 * len(all_human_tsqs)))

    df_human_test = df_human_all[df_human_all["masked_question"].isin(test_human_tsqs)]
    df_human_train = df_human_all.drop(df_human_test.index)

    # 提取训练集
    if train_mode == "chatgpt_only":
        df_train = df_chatgpt_train
        df_test = df_human_test
    elif train_mode == "human60p_only":
        df_train = df_human_train
        df_test = df_human_test
    elif train_mode == "chatgpt_with_human60p":
        df_train = pd.concat([df_chatgpt_train, df_human_train])
        df_test = df_human_test
    else:
        print("train_mode 参数不正确！", train_mode)
        exit()
    return df_train, df_test


train_df, test_df = generate_datasets_df(config["mode_train_dataset"], config["size_train_dataset"])

# 保存文件
train_df.to_csv(config["path_write_train_csv"], index=False, header=False)
test_df.to_csv(config["path_write_test_csv"], index=False, header=False)
