"""
混合检索实现
结合向量检索和关键词检索
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import Counter
from rank_bm25 import BM25Okapi
import re


class BM25Retriever:
    """BM25关键词检索器"""

    def __init__(self, documents: List[str]):
        self.documents = documents
        self.bm25 = self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        return re.findall(r'\w+', text.lower())

    def _build_index(self):
        """构建BM25索引"""
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        return BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """检索"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


class VectorRetriever:
    """向量检索器"""

    def __init__(self, documents: List[str], embedding_model: str = "BAAI/bge-large-zh"):
        from sentence_transformers import SentenceTransformer

        self.documents = documents
        self.encoder = SentenceTransformer(embedding_model)
        self.embeddings = self.encoder.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """检索"""
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(query_embedding, self.embeddings.T)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


class HybridRetriever:
    """混合检索器"""

    def __init__(self, documents: List[str], embedding_model: str = "BAAI/bge-large-zh"):
        self.documents = documents
        self.bm25_retriever = BM25Retriever(documents)
        self.vector_retriever = VectorRetriever(documents, embedding_model)

    def search(
        self,
        query: str,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        top_k: int = 5
    ) -> List[Dict]:
        """
        混合检索

        Args:
            query: 查询文本
            vector_weight: 向量检索权重
            keyword_weight: 关键词检索权重
            top_k: 返回结果数量
        """
        # 并行检索
        vector_results = self.vector_retriever.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        # 归一化分数
        def normalize(results: List[Tuple[int, float]]) -> Dict[int, float]:
            if not results:
                return {}
            scores = [s for _, s in results]
            max_s, min_s = max(scores), min(scores)
            if max_s == min_s:
                return {idx: 1.0 for idx, _ in results}
            return {idx: (s - min_s) / (max_s - min_s) for idx, s in results}

        vector_scores = normalize(vector_results)
        bm25_scores = normalize(bm25_results)

        # 收集所有候选
        all_candidates = set(vector_scores.keys()) | set(bm25_scores.keys())

        # 加权融合
        fused_scores = {}
        for doc_id in all_candidates:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            fused_scores[doc_id] = v_score * vector_weight + b_score * keyword_weight

        # 排序并返回
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:top_k]

        return [
            {
                "id": int(doc_id),
                "text": self.documents[doc_id],
                "score": float(score),
                "type": "hybrid"
            }
            for doc_id, score in top_results
        ]

    def rrf_fusion(
        self,
        query: str,
        top_k: int = 5,
        k: int = 60
    ) -> List[Dict]:
        """
        RRF融合

        RRF(d) = Σ (1 / (k + rank_i(d)))
        """
        # 获取排名
        vector_results = self.vector_retriever.search(query, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        rrf_scores = {}

        for rank, (doc_id, _) in enumerate(vector_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(bm25_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_results = sorted_results[:top_k]

        return [
            {
                "id": int(doc_id),
                "text": self.documents[doc_id],
                "score": float(score),
                "type": "rrf"
            }
            for doc_id, score in top_results
        ]


def demo():
    """演示"""
    # 知识库
    documents = [
        "人工智能是计算机科学的重要分支",
        "机器学习让计算机从数据中学习",
        "深度学习使用多层神经网络",
        "Transformer架构广泛应用于NLP",
        "GPT是OpenAI开发的大语言模型",
        "BERT是一种预训练语言模型",
        "RAG结合检索和生成两种技术",
        "向量数据库用于存储和检索向量",
        "关键词检索基于词频统计",
        "混合检索结合多种检索方法"
    ]

    # 创建混合检索器
    retriever = HybridRetriever(documents)

    # 查询
    query = "深度学习和机器学习有什么关系？"

    print(f"查询: {query}\n")

    # 混合检索
    results = retriever.search(query, vector_weight=0.6, keyword_weight=0.4, top_k=3)
    print("=== 混合检索结果 ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] {r['text']}")

    print()

    # RRF融合
    results = retriever.rrf_fusion(query, top_k=3)
    print("=== RRF融合结果 ===")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['score']:.4f}] {r['text']}")


if __name__ == "__main__":
    demo()
