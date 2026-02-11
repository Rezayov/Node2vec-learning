
import networkx as nx
from node2vec import Node2Vec

G = nx.fast_gnp_random_graph(n=10, p=0.3, seed=42)

print("Nodes:", G.nodes())
print("Edges:", G.edges())

deepwalk_dimensions = 4
deepwalk_walk_length = 5
deepwalk_num_walks = 20
deepwalk = Node2Vec(
    G,
    dimensions=deepwalk_dimensions,
    walk_length=deepwalk_walk_length,
    num_walks=deepwalk_num_walks,
    p=1,
    q=1,
    workers=1,
    seed=42
)

deepwalk_model = deepwalk.fit(
    window=3,
    min_count=1,
    batch_words=16
)

print("\n=== DeepWalk-style embedding for node 0 ===")
print(deepwalk_model.wv[str(0)])   # node ids are stored as strings



node2vec_dimensions = 4
node2vec_walk_length = 5
node2vec_num_walks = 20

node2vec = Node2Vec(
    G,
    dimensions=node2vec_dimensions,
    walk_length=node2vec_walk_length,
    num_walks=node2vec_num_walks,
    p=0.5,
    q=2.0, 
    workers=1,
    seed=42
)

node2vec_model = node2vec.fit(
    window=3,
    min_count=1,
    batch_words=16
)

print("\n=== node2vec-style embedding for node 0 ===")
print(node2vec_model.wv[str(0)])



from numpy import dot
from numpy.linalg import norm

def cosine_sim(vec_a, vec_b):
    return dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

e0_dw = deepwalk_model.wv[str(0)]
e1_dw = deepwalk_model.wv[str(1)]

e0_n2v = node2vec_model.wv[str(0)]
e1_n2v = node2vec_model.wv[str(1)]

print("\n=== Cosine similarity between node 0 and 1 ===")
print("DeepWalk-style :", cosine_sim(e0_dw,  e1_dw))
print("node2vec-style :", cosine_sim(e0_n2v, e1_n2v))
