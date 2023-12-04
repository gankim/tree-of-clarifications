import numpy as np

def rerank(reranker, query, passages, k):
    passages_cs_scores = reranker(query, passages)
    passages_cs_scores_sorted = np.argsort(passages_cs_scores)[::-1]
    passages = [passages[idx] for idx in passages_cs_scores_sorted]
    
    return passages[:k]