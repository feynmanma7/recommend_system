# Performance

|Model|Recall@500|Note|
|-----|----------|----|
|faiss|0.1842|MF,emb=32,"IDMap,Flat"|
|faiss|0.1026|MF,emb=32,"IndexIVFFlat",to tune|
|annoy|0.2040|MF,emb=32,n_trees=10|
