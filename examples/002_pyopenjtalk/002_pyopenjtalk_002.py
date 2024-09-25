# %%
from datasets import load_from_disk

# %%
dataset = load_from_disk('./wikipedia_ja_20240801')

# %%

# dataset のうち，ランダムにな2_000件をvalidation，別のランダムな2_000件をtest，その他のデータをtrainに分割
dataset_dict = dataset.shuffle(seed=42).train_test_split(test_size=4_000, shuffle=True)
train_dataset = dataset_dict['train']
valid_and_test_dataset_dict = dataset_dict['test'].train_test_split(test_size=2_000, shuffle=True)
valid_dataset = valid_and_test_dataset_dict['train']
test_dataset = valid_and_test_dataset_dict['test']

# 改めてDatasetDictを作成
from datasets import DatasetDict
dataset = DatasetDict({'train': train_dataset, 'validation': valid_dataset, 'test': test_dataset})

# %%
dataset_ = dataset.select_columns(["input_ids", "attention_mask", "labels"])

# %%
from transformers import MBartForConditionalGeneration, AutoTokenizer

model_name = "ku-nlp/bart-base-japanese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# %%
from transformers import pipeline

generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_new_tokens=60)
generated = generator(dataset['train'][0]['text'])
print(generated)

# %%
dataset["train"][0]["text"]

# %%
print(tokenizer.decode(dataset["train"][0]['input_ids']))

# %%
print(tokenizer.decode(dataset["train"][0]["labels"]))

# import sys
# sys.exit()

# %%
exp_name = "mbart_japanese_wikipedia"
output_dir = f"exp/{exp_name}/results"
logging_dir = f"exp/{exp_name}/logs"

# %%
from transformers import TrainingArguments

training_args = TrainingArguments(
    num_train_epochs=3,             # 最大3エポックとする
    per_device_train_batch_size=8,  # バッチサイズ
    auto_find_batch_size=True,      # バッチサイズを自動で見つける

    weight_decay=0.01,              # 重み減衰
    learning_rate=2e-5,             # 学習率
    warmup_steps=1000,              # ウォームアップステップ数

    evaluation_strategy="steps",    # 評価はステップごとに行う
    eval_steps=500,                 # 500ステップごとに評価を行う
    # metric_for_best_model="accuracy", # 最良のモデルの評価指標
    # greater_is_better=True,           # 評価指標が大きいほど良い場合はTrue

    output_dir=output_dir,          # モデルの保存先
    save_strategy="steps",          # モデルの保存はエポックごとに行う
    save_total_limit=3,             # 保存するモデルの数)

    logging_dir=logging_dir,        # ログの保存先
    logging_strategy="steps",       # ログの保存はエポックごとに行う
    logging_steps=500,              # 100ステップごとにログを出力する

    load_best_model_at_end=True,    # 最良のモデルを最後にロードする
)

# %%
from transformers import EarlyStoppingCallback

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3, 
    early_stopping_threshold=0.001)

# %%
from transformers import Trainer #, DataCollatorWithPadding

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_["train"],
    eval_dataset=dataset_["validation"],
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
    # callbacks=[early_stopping],
)

# %%
enable_training = True
resume_training = True
if enable_training:
    trainer.train(resume_from_checkpoint=resume_training)

model_dir=f"exp/{exp_name}/model"
model.save_pretrained(model_dir)

# # %%
# enable_model_loading = False
# output_dir = f"exp/{exp_name}/results"
# model_path = f"{output_dir}/checkpoint-58940"
# if enable_model_loading:
#     model = MBartForConditionalGeneration.from_pretrained(model_path)

# # %%
# generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# # %%
# split = 'validation'
# i = 1
# predicted = generator(dataset[split][i]["text"])
# print(f"input: {dataset[split][i]['text']}")
# print(f"target: {dataset[split][i]['phonemes']}")
# print(f"predicted: {predicted[0]['generated_text']}")

# # %%
# generator("知らない人に道を聞いてみました")


