# rag_chain.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any
from openai import OpenAI

from utils import load_env, top_k_similar
from vector_store import load_vector_store

ANSWER_MODEL = "gpt-4.1-mini"  # 과제 요구에 맞게 바꿔도 됨

class RAGChain:
    def __init__(self, k: int = 5):
        load_env()
        self.client = OpenAI()
        self.k = k
        self.embeddings, self.metadatas = load_vector_store()

    def _embed_query(self, query: str):
        resp = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        return resp.data[0].embedding

    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        q_vec = self._embed_query(query)
        scores, idxs = top_k_similar(self.embeddings, q_vec, k=self.k)

        results = []
        for score, idx in zip(scores, idxs):
            item = dict(self.metadatas[int(idx)])
            item["score"] = float(score)
            results.append(item)
        return results

    def _build_prompt(self, query: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        context_texts = []
        for c in contexts:
            src = c["metadata"].get("강좌명") or c["metadata"].get("과목코드") or c.get("id", "")
            header = f"[출처: {src}]"
            context_texts.append(header + "\n" + c["text"])

        context_block = "\n\n---\n\n".join(context_texts)

        system_msg = (
            "너는 숭실대학교 강의계획서 RAG 챗봇이야.\n"
            "아래 제공된 강의계획서 청크 내용만을 근거로 답변하고, "
            "모르는 내용은 아는 척 하지 말고 모른다고 말해."
        )

        user_msg = (
            f"다음은 강의계획서에서 추출한 관련 내용이야:\n\n"
            f"{context_block}\n\n"
            f"위 내용을 참고해서 아래 질문에 한국어로 자세히 답변해줘.\n"
            f"질문: {query}"
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def ask(self, query: str) -> Dict[str, Any]:
        contexts = self._retrieve(query)
        messages = self._build_prompt(query, contexts)

        resp = self.client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=messages,
        )
        answer = resp.choices[0].message.content

        return {
            "answer": answer,
            "contexts": contexts,
        }