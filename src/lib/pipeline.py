#! rag pipeline
from time import sleep

from lib.generate import GeneratorBase, TogetherGeneratorBase
from lib.rerank import RerankerBase
from lib.retrieval import RetrieverBase
from langchain_core.documents import Document

class RAGPipeline:
    def __init__(self, retriever, reranker, generator, k):
        self.retriever: RetrieverBase = retriever
        self.reranker: RerankerBase = reranker
        self.generator: TogetherGeneratorBase = generator
        self.k = k

    def retrieval_pass(self, question):       
        retrieved = self.retriever.query(question, k=self.k)
        reranked = self.reranker.rerank(question, retrieved)
        return {
            "retrieved": retrieved,
            "reranked": reranked,
        }

    def generation_pass(self, question: str, documents: list[Document]):
        cleaned_answer, model_output, generation_prompt= self.generator.answer_with_context(question, documents)
        return {
            "model_output": model_output,
            "cleaned_answer": cleaned_answer,
            "generation_prompt": generation_prompt, 
        }


    def detailed_pass(self, query):# -> tuple[Any, Any, Any]:
        assert False 
        if self.hypothetical_generation:
            stops = stopwords.words("english")
            query_tok = [word.lower() for word in word_tokenize(query) if word.lower() not in stops]
            if len(query_tok) < 3:
                docs = self.generator.augment_query(query)
                new_query = query + "\n" + docs
        retrieved = self.retriever.query(query if not self.hypothetical_generation else new_query, k=self.k)
        reranked = self.reranker.rerank(retrieved)
        answer = self.generator.answer(query, reranked)
        return answer, retrieved, reranked

    def __call__(self, query):
        answer, retrieved, reranked = self.detailed_pass(query)
        return answer