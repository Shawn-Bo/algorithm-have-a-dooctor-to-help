from functools import partial
from pathlib import Path

import jieba
import yaml
from transformers import BertTokenizer

with next(Path(__file__).parent.glob("config_shared.yaml")).open(mode="r") as stream:
    config = yaml.safe_load(stream)

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens

T5_pegasus_tokenizer = T5PegasusTokenizer.from_pretrained(
    config["model_name"],
    model_max_length=64,
    cache_dir=f"{config['model_base']}/{config['model_name']}")

T5_pegasus_tokenizer.add_tokens(["【-】", "【?】"])

if __name__ == "__main__":
    print(T5_pegasus_tokenizer.encode("你好世界，我不是很好！"))
    