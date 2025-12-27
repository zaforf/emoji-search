from flask import Flask, request
import pandas as pd
import torch
import time
from sentence_transformers import SentenceTransformer, util

emojis = pd.read_parquet("data/cleaned_emojis.parquet")
text_embeddings = torch.load("data/text_embeddings.pt")
image_embeddings = torch.load("data/image_embeddings.pt")
text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = SentenceTransformer('clip-ViT-B-32')

app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    alpha = float(request.args.get('alpha', 0.5))
    if not query:
        return {'results': [], 'timing': []}

    timing = []

    t0 = time.perf_counter()
    q_txt = text_model.encode(query, convert_to_tensor=True)
    timing.append({'name': 'text_encode', 'ms': (time.perf_counter() - t0) * 1000})

    t0 = time.perf_counter()
    scores_txt = util.cos_sim(q_txt, text_embeddings)[0]
    timing.append({'name': 'text_cos_sim', 'ms': (time.perf_counter() - t0) * 1000})

    t0 = time.perf_counter()
    q_vis = clip_model.encode(query, convert_to_tensor=True)
    timing.append({'name': 'clip_encode', 'ms': (time.perf_counter() - t0) * 1000})

    t0 = time.perf_counter()
    scores_vis = util.cos_sim(q_vis, image_embeddings)[0]
    timing.append({'name': 'clip_cos_sim', 'ms': (time.perf_counter() - t0) * 1000})

    t0 = time.perf_counter()
    combined_scores = alpha * scores_txt + (1 - alpha) * scores_vis
    top_scores, top_indices = torch.topk(combined_scores, k=10)
    timing.append({'name': 'top_k', 'ms': (time.perf_counter() - t0) * 1000})

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

    return {'results': results, 'timing': timing}

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(debug=True)