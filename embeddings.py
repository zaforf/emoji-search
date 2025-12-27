import os
import urllib.request
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import PIL.Image

SKIN_TONE_MODIFIERS = ['1F3FB', '1F3FC', '1F3FD', '1F3FE', '1F3FF']
SPRITE_SHEET_URL = "https://cdn.jsdelivr.net/npm/emoji-datasource-twitter@16.0.0/img/twitter/sheets/64.png"
SPRITE_SHEET_PATH = "data/sheet_twitter_64.png"
INDEX_PATH = "data/emoji.json"
DATASET_PATH = "hf://datasets/badrex/LLM-generated-emoji-descriptions/data/train-00000-of-00001.parquet"

def download_sprite_sheet():
    if not os.path.exists(SPRITE_SHEET_PATH):
        print("downloading sprite sheet...")
        urllib.request.urlretrieve(SPRITE_SHEET_URL, SPRITE_SHEET_PATH)

def clean_dataset(df):
    # filter out skin tone modifiers
    mask_skin_tone = df['unicode'].apply(lambda u: any(mod in u for mod in SKIN_TONE_MODIFIERS))
    df = df[~mask_skin_tone].copy()

    df = df.drop_duplicates(subset=['short description'])
    return df

def generate_text_embeddings(df, model_name='all-MiniLM-L6-v2'):
    prompts = df["short description"] + ", associated with: " + df["tags"].str.join(", ")

    print(f"generating text embeddings for {len(prompts)} items...")
    model = SentenceTransformer(model_name)
    return model.encode(prompts.tolist(), convert_to_tensor=True)

def generate_image_embeddings(df, model_name='clip-ViT-B-32'):
    download_sprite_sheet()
    emoji_index = pd.read_json(INDEX_PATH)
    sheet = PIL.Image.open(SPRITE_SHEET_PATH)
    white_bg = PIL.Image.new("RGB", (64, 64), (255, 255, 255))

    coord_map = emoji_index.groupby('unified')[['sheet_x', 'sheet_y']].first().to_dict('index')
    non_qual_map = emoji_index.groupby('non_qualified')[['sheet_x', 'sheet_y']].first().to_dict('index')
    coord_map.update(non_qual_map)

    images = []
    valid_indices = []

    for idx, row in df.iterrows():
        code = row['unicode'][2:].replace(' ', '-')

        coords = coord_map.get(code)
        if coords:
            # find in sprite sheet
            sheet_size = 66 # 64px + 2px padding
            left = coords['sheet_x'] * sheet_size + 1
            upper = coords['sheet_y'] * sheet_size + 1
            crop = sheet.crop((left, upper, left + 64, upper + 64))

            # give white background
            emoji_image = white_bg.copy()
            emoji_image.paste(crop, mask=crop.split()[3])

            images.append(emoji_image)
            valid_indices.append(idx)

    print(f"generating image embeddings for {len(images)}/{len(df)} items...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(images, convert_to_tensor=True)

    return embeddings, valid_indices

if __name__ == '__main__':
    df = pd.read_parquet(DATASET_PATH)
    df = clean_dataset(df)
    img_embeddings, valid_indices = generate_image_embeddings(df)
    df = df.loc[valid_indices].copy()
    text_embeddings = generate_text_embeddings(df)

    torch.save(img_embeddings, "data/image_embeddings.pt")
    torch.save(text_embeddings, "data/text_embeddings.pt")
    df.to_parquet("data/cleaned_emojis.parquet")
