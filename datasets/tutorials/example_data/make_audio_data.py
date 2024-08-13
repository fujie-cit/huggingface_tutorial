# 1秒間のランダムな8kHzでサンプリングされた音声データ（wavファイル）を生成し，
# 下記の形で保存する
# audio/pokemon/train/grass/bulbasaur.wav
# audio/pokemon/train/fire/charmander.wav
# audio/pokemon/train/water/squirtle.wav
# audio/pokemon/test/grass/ivysaur.wav
# audio/pokemon/test/fire/charmeleon.wav
# audio/pokemon/train/water/squirtle.wav

import os
import numpy as np
import scipy.io.wavfile as wav

# データセットの保存先
dataset_dir = 'audio/pokemon'
train_dir = os.path.join(dataset_dir, 'train')
test_dir = os.path.join(dataset_dir, 'test')

# ディレクトリの作成
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# データセットのラベル
labels = ['grass', 'fire', 'water']

# データセットの分割
train_data = [
    ('grass', 'bulbasaur.wav'),
    ('fire', 'charmander.wav'),
    ('water', 'squirtle.wav')
]

test_data = [
    ('grass', 'ivysaur.wav'),
    ('fire', 'charmeleon.wav'),
    ('water', 'wartortle.wav')
]

# データセットの作成
fs = 8000
duration = 1
for label, filename in train_data:
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    data = np.random.randn(fs * duration)
    wav.write(os.path.join(train_dir, label, filename), fs, data)

for label, filename in test_data:
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)
    data = np.random.randn(fs * duration)
    wav.write(os.path.join(test_dir, label, filename), fs, data)
