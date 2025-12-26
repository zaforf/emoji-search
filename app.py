from flask import Flask, request
import pandas as pd
from sentence_transformers import SentenceTransformer, util

emojis = pd.read_parquet("hf://datasets/badrex/LLM-generated-emoji-descriptions/data/train-00000-of-00001.parquet")
app = Flask(__name__)

@app.route('/search')
def search():
    query = request.args.get('query', '')
    if not query:
        return {'results': []}

    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, embeddings, top_k=10)[0]

    results = []
    for hit in hits:
        emoji_data = emojis.iloc[hit['corpus_id']]
        results.append({
            'emoji': emoji_data['character'],
            'name': emoji_data['short description'],
            'score': hit['score']
        })

    return {'results': results}

@app.route('/')
def index():
    return app.send_static_file('index.html')

def form_prompts():
    result = []
    for index, row in emojis.iterrows():
        # Create prompt combining description and tags
        prompt = f"{row["short description"]}, associated with: {", ".join(row["tags"])}"
        result.append(prompt)
    return result

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    prompts = form_prompts()
    embeddings = model.encode(prompts, convert_to_tensor=True)
    app.run(debug=True)