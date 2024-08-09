from __future__ import unicode_literals
# %%
from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p_prosody

text = "私の名前は藤江と言いますか？"
phoneme_list = pyopenjtalk_g2p_prosody(text)
# phoneme_list内の要素の "N" を "nn" に変換
phoneme_list = [phoneme.replace("N", "nn") for phoneme in phoneme_list]
phonemes = " ".join(phoneme_list)

print(text)
print(phonemes)

# %%
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "sonoisa/t5-base-japanese"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# %%
from transformers import pipeline

generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
generated = generator(phonemes)
print(generated)

# %%
from datasets import load_dataset

dataset = load_dataset("shunk031/livedoor-news-corpus")

# %%
# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja から引用・一部改変
import re
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]+', '〜', s)  # normalize tildes (modified by Isao Sonobe)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

# %%
def preprocess(example):
    text = example["title"]
    text = text.replace("\t", " ")
    text = text.strip()
    text = normalize_neologd(text)
    text = text.lower()
    phoneme_list = pyopenjtalk_g2p_prosody(text)
    phoneme_list = [phoneme.replace("N", "nn") for phoneme in phoneme_list]
    phonemes = " ".join(phoneme_list)
    example["text"] = text
    example["phonemes"] = phonemes
    return example

dataset = dataset.map(preprocess)

# %%
def tokenize_function(examples):
    model_inputs = tokenizer(
        text=examples["text"],
        max_length=model.config.max_length, # prob. 512
        padding="max_length",
        truncation=True)
    labels = tokenizer(
        text_target=examples["phonemes"],
        max_length=model.config.max_length,
        padding="max_length",
        truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    examples.update(model_inputs)
    return examples

dataset = dataset.map(tokenize_function, batched=True)

# %%
dataset["train"][0].keys()

# %%
exp_name = "base"
output_dir = f"exp/{exp_name}/results"
logging_dir = f"exp/{exp_name}/logs"

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    num_train_epochs=10,            # 最大10エポックとする
    per_device_train_batch_size=8,  # バッチサイズ
    auto_find_batch_size=True,    # バッチサイズを自動で見つける

    weight_decay=0.01,              # 重み減衰
    learning_rate=2e-5,             # 学習率
    warmup_steps=500,               # ウォームアップステップ数

    evaluation_strategy="epoch",    # 評価はエポックごとに行う
    # metric_for_best_model="accuracy", # 最良のモデルの評価指標
    # greater_is_better=True,           # 評価指標が大きいほど良い場合はTrue

    output_dir=output_dir,          # モデルの保存先
    save_strategy="epoch",          # モデルの保存はエポックごとに行う
    save_total_limit=3,             # 保存するモデルの数)

    logging_dir=logging_dir,        # ログの保存先
    logging_strategy="steps",       # ログの保存はエポックごとに行う
    logging_steps=100,              # 100ステップごとにログを出力する

    load_best_model_at_end=True,    # 最良のモデルを最後にロードする
)

# %%
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3, 
    early_stopping_threshold=0.001)

# %%
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    # compute_metrics=compute_metrics,
    # callbacks=[early_stopping],
)

# %%
trainer.train()

# %%
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# %%
i = 0
predicted = generator(dataset["train"][i]["text"])
print(f"input: {dataset['train'][i]['text']}")
print(f"target: {dataset['train'][i]['phonemes']}")
print(f"predicted: {predicted[0]['generated_text']}")

# %%
tokenizer.decode(dataset["train"][0]["input_ids"])


