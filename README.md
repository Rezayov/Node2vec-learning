# 🔬 DeepWalk & node2vec – Graph Embedding Demonstration

## 🎯 Project Overview

This project contains two Python scripts that demonstrate **graph embedding** techniques using the `node2vec` library. They generate a random graph, compute low‑dimensional vector representations (embeddings) for each node, and compare the results of **DeepWalk** (uniform random walks) vs. **node2vec** (biased random walks controlled by parameters `p` and `q`).

> **Learning goal:** Understand how random walks on graphs can be used to create node embeddings, how hyperparameters affect the embedding space, and how to visualise and compare embeddings with cosine similarity.

---

## 📁 File Breakdown

| File | Purpose | Key Concepts |
|------|---------|---------------|
| `DEEPWALK.py` | Full demonstration with 2D embeddings and scatter plots | Graph generation, DeepWalk, node2vec, 2D projection, visual comparison |
| `NODE2VEC.py` | Focused on 4‑dimensional embeddings and cosine similarity | Higher‑dim embeddings, cosine similarity, quantitative comparison |

---

## 🧠 Core Concepts Covered

### 1. **Graph Representation**
- `networkx` is used to generate a random graph (`fast_gnp_random_graph`).
- Nodes are unlabeled (just integers); edges are random with probability `p=0.3`.

### 2. **Random Walks on Graphs**
- **DeepWalk**: uniform random walks (equivalent to `p=1, q=1`). Treats the graph as a set of sentences (walks) and learns node embeddings using Word2Vec (skip‑gram).
- **node2vec**: biased random walks with two parameters:
  - `p` (return parameter): controls likelihood of immediately revisiting a node.
  - `q` (in‑out parameter): controls BFS vs DFS‑like exploration.

### 3. **Embedding Generation**
- `Node2Vec` class from the `node2vec` library:
  - `dimensions`: size of the output vector.
  - `walk_length`: length of each random walk.
  - `num_walks`: number of walks per node.
  - `p`, `q`: bias parameters.
  - `workers`: parallel threads.
- `fit()` uses Word2Vec (via `gensim`) to learn embeddings.

### 4. **Cosine Similarity**
- Measures similarity between two node embeddings (independent of magnitude).
- `cosine_sim(a,b) = (a·b) / (||a|| * ||b||)`

### 5. **Visualisation**
- For 2‑dimensional embeddings (`DEEPWALK.py`), scatter plots show how nodes are positioned in the learned latent space.
- Labels are added to identify each node.

---

## 🛠️ Skills You Will Develop

After studying and running these scripts, you will be able to:

1. **Generate random graphs** with `networkx`.
2. **Install and use the `node2vec` library** (compatible with `gensim`).
3. **Explain the difference** between DeepWalk and node2vec (bias parameters).
4. **Tune `p` and `q`** to encourage different walk behaviours.
5. **Compute and interpret cosine similarity** between node embeddings.
6. **Visualise embeddings** in 2D using `matplotlib`.
7. **Understand that node embeddings are stored** as `model.wv[str(node)]`.
8. **Compare embedding spaces** qualitatively (plots) and quantitatively (similarity scores).

---

## 🧪 How to Run the Code

### Prerequisites
```bash
pip install networkx node2vec matplotlib numpy
```

### Run the demonstration
```bash
python DEEPWALK.py
```
This will:
1. Create a random graph with 10 nodes.
2. Show the original graph.
3. Compute 2‑D DeepWalk embeddings and plot them.
4. Compute 2‑D node2vec embeddings (`p=0.5, q=2.0`) and plot them.

```bash
python NODE2VEC.py
```
This will:
1. Generate the same random graph.
2. Compute 4‑D DeepWalk embeddings and print node 0’s vector.
3. Compute 4‑D node2vec embeddings and print node 0’s vector.
4. Calculate and print cosine similarity between node 0 and node 1 for both methods.

---

## 📊 Expected Output (Qualitative)

- **DeepWalk** (uniform walks) tends to produce more globally distributed embeddings.
- **node2vec with `p<1, q>1`** favours local, BFS‑like walks – nodes in dense clusters may be closer in embedding space.
- Cosine similarity values will differ because the two methods capture different structural roles.

---

## 🔍 Real‑World Applications

Graph embeddings are used for:
- **Social network analysis** – friend recommendations, community detection.
- **Knowledge graphs** – link prediction, entity resolution.
- **Bioinformatics** – protein‑protein interaction networks.
- **Recommendation systems** – item‑item graphs.
- **Fraud detection** – suspicious node identification.

---

## 🧠 Further Exploration

- Change graph size (`n`) and density (`p`).
- Vary `p` and `q` to see how the embedding space changes (e.g., `p=2, q=0.5` encourages DFS).
- Increase `dimensions` to 8 or 16 and use PCA/t‑SNE for visualisation.
- Compare against other embedding methods (e.g., `node2vec` vs `GraphSAGE`).
- Use embeddings for a downstream task like **node classification** (with `sklearn`).

---

## 📚 Dependencies

| Library | Purpose |
|---------|---------|
| `networkx` | Graph creation and manipulation |
| `node2vec` | Random walk generation + Word2Vec embedding |
| `matplotlib` | Plotting original graph and embedding scatter |
| `numpy` | Cosine similarity calculation |

---

## ⚠️ Notes

- The `node2vec` library internally uses `gensim`; if not installed, it will be pulled as a dependency.
- Node IDs are stored as **strings** in the Word2Vec model: `model.wv[str(node)]`.
- The random graph is generated with `seed=42` so results are reproducible.

---

*These scripts provide a hands‑on introduction to one of the most influential graph representation learning techniques – turning nodes into vectors while preserving structural information.*
