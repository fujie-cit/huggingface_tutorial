# 64x64のランダムなカラー画像を生成し，次のように保存する
# image/pokemon/train/grass/bulbasaur.png
# image/pokemon/train/fire/charmander.png
# image/pokemon/train/water/squirtle.png
# image/pokemon/test/grass/ivysaur.png
# image/pokemon/test/fire/charmeleon.png
# image/pokemon/test/water/wartortle.png

import os
import numpy as np
from PIL import Image

# 乱数のシードを設定
np.random.seed(0)

# 画像サイズ
size = (64, 64)

# 画像の色
colors = {
    'grass': [0, 128, 0],
    'fire': [255, 0, 0],
    'water': [0, 0, 255]
}

# 画像の保存先
save_dir = 'image/pokemon'

# 画像を生成する関数
def generate_image(color):
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    for i in range(3):
        image[:, :, i] = color[i]
    return Image.fromarray(image)

# 画像を保存する関数
def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)

# 画像を生成して保存
for i, (name, color) in enumerate(colors.items()):
    for j, (dir_name, data_type) in enumerate([('train', 'grass'), ('test', 'fire')]):
        image = generate_image(color)
        save_image(image, os.path.join(save_dir, dir_name, name, f'{name}_{j + 1}.png'))

                                                                    
                                                