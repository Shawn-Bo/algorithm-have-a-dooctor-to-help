# 基于中文BERT的提问意图分类实验设计

## 测试不同规模ChatGPT生成数据集对模型性能的影响

相关参数：

- train_mode
  - chatgpt_only
  - human_only
  - （每个实验重复5轮取平均）

## 比较ChatGPT生成数据集和人工标注生成数据集的性能差距

- train_size
  - 1k
  - 5k
  - 10k
  - 20k
  - 30k
  - 40k
  - 50k
  - 60k
  - 70k
  - 80k

## 路径参数

- path_cache_dir
- path_xxxx

## 工作参数

- job0_gen_data
- job1_train
- job2_test


