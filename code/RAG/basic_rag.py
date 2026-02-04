"""
基础RAG实现示例
基于文本嵌入的检索增强生成系统
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

# 配置
EMBEDDING_MODEL = "BAAI/bge-large-zh"
TOP_K = 5


class VectorStore:
    """简单的向量存储（使用numpy）"""

    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.documents = []
        self.embeddings = np.array([])

    def add(self, texts: List[str], embeddings: np.ndarray):
        """添加文档"""
        self.documents.extend(texts)
        if len(self.embeddings) == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

    def search(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> List[Tuple[int, float]]:
        """搜索"""
        scores = np.dot(query_embedding, self.embeddings.T)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def save(self, path: str):
        """保存"""
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        with open(os.path.join(path, "documents.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def load(self, path: str):
        """加载"""
        self.embeddings = np.load(os.path.join(path, "embeddings.npy"))
        with open(os.path.join(path, "documents.json"), "r", encoding="utf-8") as f:
            self.documents = json.load(f)


class BasicRAG:
    """基础RAG系统"""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        self.encoder = SentenceTransformer(embedding_model)
        self.vector_store = VectorStore()
        self.history = []

    def build_index(self, documents: List[str], save_path: str = None):
        """构建索引"""
        embeddings = self.encoder.encode(documents, normalize_embeddings=True)
        self.vector_store.add(documents, embeddings)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.vector_store.save(save_path)

        print(f"已构建索引，包含 {len(documents)} 篇文档")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
        """检索"""
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]
        results = self.vector_store.search(query_embedding, top_k)

        return [
            (self.vector_store.documents[idx], score)
            for idx, score in results
        ]

    def generate(self, query: str, contexts: List[str]) -> str:
        """
        基于检索结果生成回答

        注意：实际项目中应调用LLM API或本地模型
        这里使用简单的模板演示
        """
        # 构建prompt
        context_text = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(contexts)])

        prompt = f"""请根据以下检索到的信息回答问题。如果信息不足以回答，请说明。

检索到的相关信息：
{context_text}

问题：{query}

回答："""

        # 实际项目中这里调用LLM
        # response = llm.generate(prompt)
        # return response

        # 演示用：简单返回
        return f"根据检索到的信息，关于'{query}'的回答是：..."

    def chat(self, query: str) -> str:
        """对话"""
        # 检索
        contexts = self.retrieve(query)

        # 生成
        response = self.generate(query, [c[0] for c in contexts])

        # 记录历史
        self.history.append({"query": query, "response": response})

        return response


def demo():
    """演示"""
    # 初始化
    rag = BasicRAG()

    # 知识库文档
    documents = [
        "人工智能是计算机科学的一个分支，致力于创建智能机器。",
        "机器学习是人工智能的一个子领域，让计算机通过数据学习。",
        "深度学习是机器学习的一个重要分支，使用多层神经网络。",
        "Transformer是一种神经网络架构，广泛用于NLP任务。",
        "RAG（检索增强生成）结合了检索和生成两种技术。",
        "大语言模型如GPT、Llama等在各类任务上表现出色。",
        "自然语言处理研究如何让计算机理解和生成人类语言。"
    ]

    # 构建索引
    rag.build_index(documents, save_path="./data/rag_index")

    # 对话
    while True:
        query = input("\n请输入问题 (输入'quit'退出): ")
        if query.lower() == "quit":
            break

        response = rag.chat(query)
        print(f"\n回答: {response}")

        # 显示检索结果
        contexts = rag.retrieve(query)
        print("\n检索到的相关文档:")
        for i, (ctx, score) in enumerate(contexts, 1):
            print(f"  {i}. [{score:.4f}] {ctx}")


if __name__ == "__main__":
    demo()
