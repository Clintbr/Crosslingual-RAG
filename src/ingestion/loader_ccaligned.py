from datasets import load_dataset
from tqdm import tqdm
import os

LIMIT = 1000

print("Loading OPUS Books...")

dataset = load_dataset(
    "opus_books",
    "de-fr",
    split="train",
    streaming=True
)

dataset_en = load_dataset(
    "opus_books",
    "de-en",
    split="train",
    streaming=True
)

output_dir = r"/multilingual/multi_datasets/ccaligned_dataset"

os.makedirs(output_dir, exist_ok=True)

de_texts = []
fr_texts = []
en_texts = []

file_index = 1
file_index_en = 1

for i, item in enumerate(tqdm(dataset)):
    print('item number' + i.__str__())
    de = item["translation"]["de"]
    fr = item["translation"]["fr"]

    if de:
        de_texts.append(de)

    if fr:
        fr_texts.append(fr)

    if (i + 1) % LIMIT == 0:
        print('Speicherung Nummer:' + i.__str__())
        de_full = "\n".join(de_texts)
        fr_full = "\n".join(fr_texts)

        de_path = os.path.join(output_dir, f"de{file_index}.txt")
        fr_path = os.path.join(output_dir, f"fr{file_index}.txt")

        with open(de_path, "w", encoding="utf-8") as f:
            f.write(de_full)

        with open(fr_path, "w", encoding="utf-8") as f:
            f.write(fr_full)

        print(f"\nSaved:")
        print(de_path)
        print(fr_path)

        # Reset
        de_texts = []
        fr_texts = []

        file_index += 1

    if i == 10 * LIMIT - 1:
        break


for i, item in enumerate(tqdm(dataset_en)):
    print('item number' + i.__str__())
    en = item["translation"]["en"]

    if en:
        en_texts.append(en)

    if (i + 1) % LIMIT == 0:
        print('Speicherung Nummer:' + i.__str__())
        en_full = "\n".join(en_texts)

        en_path = os.path.join(output_dir, f"en{file_index_en}.txt")

        with open(en_path, "w", encoding="utf-8") as f:
            f.write(en_full)

        print(f"\nSaved:")
        print(en_path)

        # Reset
        en_texts = []

        file_index_en += 1

    if i == 10 * LIMIT - 1:
        break

if de_texts or fr_texts or en_texts:

    de_full = "\n".join(de_texts)
    fr_full = "\n".join(fr_texts)
    en_full = "\n".join(en_texts)

    de_path = os.path.join(output_dir, f"de{file_index}.txt")
    fr_path = os.path.join(output_dir, f"fr{file_index}.txt")
    en_path = os.path.join(output_dir, f"en{file_index_en}.txt")

    with open(de_path, "w", encoding="utf-8") as f:
        f.write(de_full)

    with open(fr_path, "w", encoding="utf-8") as f:
        f.write(fr_full)

    with open(en_path, "w", encoding="utf-8") as f:
        f.write(en_full)

    print(f"\nSaved remaining:")
    print(de_path)
    print(fr_path)
    print(en_path)