import os
import glob
from flask import Flask, request
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

emojis = pd.read_parquet("data/cleaned_emojis.parquet")
text_embeddings = torch.load("data/text_embeddings.pt")
image_embeddings = torch.load("data/image_embeddings.pt")
text_model = SentenceTransformer('all-MiniLM-L6-v2')

model_id = "openai/clip-vit-base-patch32"
model_folder = "./tiny_clip/"
safetensors_path = os.path.join(model_folder, "model.safetensors")

if not os.path.exists(safetensors_path):
    print("reconstructing model from chunks...")
    chunk_pattern = os.path.join(model_folder, "chunk_*")
    chunks = sorted(glob.glob(chunk_pattern))
    
    with open(safetensors_path, "wb") as f_out:
        for chunk_path in chunks:
            with open(chunk_path, "rb") as f_in:
                f_out.write(f_in.read())
    print(f"model rebuilt: {os.path.getsize(safetensors_path)} bytes")

clip_tokenizer = CLIPTokenizer.from_pretrained("./tiny_clip")
clip_model = CLIPTextModelWithProjection.from_pretrained("./tiny_clip")

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    alpha = float(request.args.get('alpha', 0.5))
    if not query:
        return {'results': []}

    # encode and search
    q_txt = text_model.encode(query, convert_to_tensor=True)
    scores_txt = util.cos_sim(q_txt, text_embeddings)[0]

    inputs = clip_tokenizer(query, return_tensors="pt")
    with torch.no_grad():
        text_outputs = clip_model(**inputs)
        q_vis = text_outputs.text_embeds
    scores_vis = util.cos_sim(q_vis, image_embeddings)[0]

    combined_scores = alpha * scores_txt + (1 - alpha) * scores_vis
    top_scores, top_indices = torch.topk(combined_scores, k=10)

    results = []
    for score, idx in zip(top_scores, top_indices):
        idx = idx.item()
        emoji_data = emojis.iloc[idx]
        results.append({
            'emoji': emoji_data['character'],
            'name': emoji_data['short description'],
            'score': float(score),
            'score_txt': float(scores_txt[idx]),
            'score_vis': float(scores_vis[idx])
        })

    return {'results': results}

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)