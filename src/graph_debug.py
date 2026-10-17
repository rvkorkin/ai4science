import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.graph_rag import build_or_load_graph
from llama_index.core import Settings

index = build_or_load_graph()
store = index.property_graph_store

chunks = [n for n in store.graph.nodes.values() if type(n).__name__ == "ChunkNode" and n.embedding]
print(f"Chunks with embedding: {len(chunks)}")

E = np.array([c.embedding for c in chunks], dtype=np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)

q = np.array(Settings.embed_model.get_query_embedding("DAS strain analysis fiber optic"), dtype=np.float32)
q /= np.linalg.norm(q)

sims = E @ q
top = sims.argsort()[::-1][:5]

for rank, i in enumerate(top):
    c = chunks[i]
    m = c.properties
    print(f"\n--- {rank} score={sims[i]:.3f} ---")
    print(f"{m.get('file_name','?')} p.{m.get('page_label','?')}")
    print(c.text[:300])