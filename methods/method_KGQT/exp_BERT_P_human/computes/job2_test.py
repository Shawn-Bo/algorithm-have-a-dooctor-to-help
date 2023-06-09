# 等训练有进展了再接着做

import logging
import typer
from pathlib import Path
from typing import List
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from shared_modules.T5_modules import T5Module
from shared_modules.T5_pegasus_tokenizer import T5_pegasus_tokenizer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
        report_path: str = typer.Option("../outputs/report.log"),
        path_read_test_df: Path = typer.Option("../stores/test_df.csv", help="保存test_df.csv的路径"),
        batch_size: int = typer.Option(8),
        device: str = typer.Option("cuda:0"),




):
    # 加载配置文件
    logging.basicConfig(filename=report_path, filemode="w",
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S",
                        level=logging.DEBUG)

    list_test = pd.read_csv(path_read_test_df, names=["nlq", "tsq"]).values.tolist()
    num_test = len(list_test)
    test_batch_size = batch_size

    dict_P2id = {'患病概率': 0, '所属科室': 1, '推荐药物': 2, '推荐食谱': 3, '描述': 4, '传播方式': 5, '好评药物': 6,
                 '就诊科室': 7, '常用药物': 8, '并发症': 9, '易患人群': 10, '是否纳入医保': 11, '检查项目': 12,
                 '治愈概率': 13, '治疗方法': 14, '治疗时长': 15, '治疗费用': 16, '病因': 17, '症状': 18, '诊断检查': 19,
                 '预防方法': 20, '宜吃': 21, '忌吃': 22, '生产药品': 23, '目录': 24}


    def nlq2tsq(nlq_list):
        source_encoding = T5_pegasus_tokenizer.batch_encode_plus(
            nlq_list,
            max_length=64,
            padding="max_length",
            truncation="only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        source_encoding = source_encoding.to(device)

        result_token_ids_list = trained_model.model.generate(
            input_ids=source_encoding["input_ids"],
            attention_mask=source_encoding["attention_mask"],
            num_beams=3,
            max_length=64,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            num_return_sequences=1
        )

        tsq_list = [
            T5_pegasus_tokenizer.decode(result_token_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=True).replace(" ", "")
            for result_token_ids in result_token_ids_list]
        return tsq_list

    def TSQlist_to_Slist_Plist(tsq_list: List[str]):
        """
            将tsq列表转为S列表和P列表
        """
        S_list = []
        P_list = []
        for tsq in tsq_list:
            if tsq.count("【-】") != 2:  # 格式有问题，直接作废
                S_list.append("None")
                P_list.append(-1)
            else:  # 格式没问题
                tsq_splits = tsq.split("【-】")

                S = tsq_splits[0]
                S_list.append(S)

                P = tsq_splits[1]
                if P in dict_P2id.keys():
                    P_list.append(dict_P2id[P])
                else:
                    P_list.append(-1)
        return S_list, P_list

    ckpt_files = sorted(Path(cfg["dir_read_checkpoint"]).glob('*.ckpt'), key=lambda f: f.stat().st_mtime,
                        reverse=True)
    # 取最新的文件
    if ckpt_files:
        latest_ckpt_file = ckpt_files[0]
        logging.info(f"The latest checkpoint file is: {latest_ckpt_file}")
    else:
        logging.info("No checkpoint files found.")
        exit()
    trained_model = T5Module.load_from_checkpoint(latest_ckpt_file)
    trained_model.to(device)
    trained_model.freeze()

    actual_Pids_all = []
    predict_Pids_all = []
    actual_S_all = []
    predict_S_all = []

    for i in range(0, num_test, test_batch_size):
        test_batch = list_test[i:i + test_batch_size]

        actual_tsqs = [pair[0] for pair in test_batch]
        acutal_nlqs = [pair[1] for pair in test_batch]
        # 从实际的tsq中获取 P_list
        actual_S_list, actual_P_list = TSQlist_to_Slist_Plist(actual_tsqs)

        predict_tsqs = nlq2tsq(acutal_nlqs)
        # 从预测的 tsqs 中获取预测的 P_list
        predict_S_list, predict_P_list = TSQlist_to_Slist_Plist(predict_tsqs)

        actual_Pids_all.extend(actual_P_list)
        predict_Pids_all.extend(predict_P_list)
        actual_S_all.extend(actual_S_list)
        predict_S_all.extend(predict_S_list)

    # compute precision, recall, f1-score and support for each class
    precision, recall, f1_score, support = precision_recall_fscore_support(actual_Pids_all, predict_Pids_all,
                                                                           average=None, zero_division=1)

    # logging.info precision, recall, f1-score and support for each class
    for i in range(len(precision)):
        logging.info(
            "Class {}: Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}, Support = {}".format(i, precision[i],
                                                                                                    recall[i],
                                                                                                    f1_score[i],
                                                                                                    support[i]))

    # compute macro-averaged precision, recall, f1-score
    macro_precision, macro_recall, macro_f1_score, _ = precision_recall_fscore_support(actual_Pids_all,
                                                                                       predict_Pids_all,
                                                                                       average='macro', zero_division=1)
    logging.info(
        "Macro-averaged Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}".format(macro_precision, macro_recall,
                                                                                       macro_f1_score))

    # compute micro-averaged precision, recall, f1-score
    micro_precision, micro_recall, micro_f1_score, _ = precision_recall_fscore_support(actual_Pids_all,
                                                                                       predict_Pids_all,
                                                                                       average='micro', zero_division=1)

    logging.info(
        "Micro-averaged Precision = {:.2f}, Recall = {:.2f}, F1-score = {:.2f}".format(micro_precision, micro_recall,
                                                                                       micro_f1_score))

    # 计算S的准确率 slist和plist一定等长度
    num_correct_S = 0
    num_all_S = len(actual_S_all)
    for i in range(len(actual_S_all)):
        if actual_S_all[i].lower() == predict_S_all[i].lower():
            num_correct_S += 1

    logging.info("Accuracy of Subject Prediction = {:.3f}".format(num_correct_S / num_all_S))


if __name__ == "__main__":
    main()
