from langchain_core.documents import Document


class RerankerBase:
    def __init__(self):
        pass

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        raise NotImplementedError


# A reranker that does nothing
class NoopReranker(RerankerBase):
    def __init__(self):
        super().__init__()
    
    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        return docs
