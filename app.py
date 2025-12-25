from flask import Flask, request
import csv
from sentence_transformers import SentenceTransformer, util

emoji_csv = "emoji_df.csv"
emojis = list(csv.DictReader(open(emoji_csv)))
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
        emoji_data = emojis[hit['corpus_id']]
        results.append({
            'char': emoji_data['emoji'],
            'name': emoji_data['name'],
            'score': hit['score']
        })

    return {'results': results}

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([row['name'] for row in emojis], convert_to_tensor=True)
    app.run(debug=True)