import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np

G = nx.fast_gnp_random_graph(n=10, p=0.3, seed=42)

plt.figure()
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True)
plt.title("Original Graph (10 nodes)")
plt.show()

deepwalk = Node2Vec(
    G,
    dimensions=2,
    walk_length=5,
    num_walks=20,
    p=1,
    q=1,
    workers=1,
    seed=42
)
deepwalk_model = deepwalk.fit(window=3, min_count=1, batch_words=16)

nodes = sorted(G.nodes())
emb_dw = np.array([deepwalk_model.wv[str(n)] for n in nodes])

plt.figure()
plt.scatter(emb_dw[:, 0], emb_dw[:, 1])
for i, n in enumerate(nodes):
    plt.text(emb_dw[i, 0], emb_dw[i, 1], str(n))
plt.title("DeepWalk-style Embedding Space (p=1, q=1)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.axhline(0)
plt.axvline(0)
plt.show()

# 4) Node2vec-style embeddings (biased walks)
node2vec = Node2Vec(
    G,
    dimensions=2,
    walk_length=5,
    num_walks=20,
    p=0.5,     # encourage returning
    q=2.0,     # more BFS-like, stay local
    workers=1,
    seed=42
)
node2vec_model = node2vec.fit(window=3, min_count=1, batch_words=16)

emb_n2v = np.array([node2vec_model.wv[str(n)] for n in nodes])

plt.figure()
plt.scatter(emb_n2v[:, 0], emb_n2v[:, 1])
for i, n in enumerate(nodes):
    plt.text(emb_n2v[i, 0], emb_n2v[i, 1], str(n))
plt.title("node2vec-style Embedding Space (p=0.5, q=2.0)")
plt.xlabel("dim 1")
plt.ylabel("dim 2")
plt.axhline(0)
plt.axvline(0)
plt.show()
