# %%
import os
# os.environ["HF_HUB_CACHE"] = "/autofs/diamond3/share/cache/huggingface"
os.environ["HF_HOME"] = "/autofs/diamond3/share/cache/huggingface"

# %%
from datasets import load_dataset

# %%
dataset = load_dataset("wikipedia", language="ja", date="20240801")

# %%



