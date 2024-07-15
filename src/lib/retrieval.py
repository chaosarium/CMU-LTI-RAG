#! retrieval system
from typing import Any, List, Optional, Sequence
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import warnings
from pprint import pprint
warnings.filterwarnings(action="ignore", message=".*a chunk of size.*")
import os
import sys
from ragatouille import RAGPretrainedModel
from lib.preprocess import DatasetBase
import langchain_community
from langchain_core.retrievers import BaseRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

class RetrieverBase:
    def __init__(self) -> None:
        
        self.docs: list[Document]
        pass
    
    def query(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        raise NotImplementedError

class BM25Retriever(RetrieverBase):
    def __init__(
        self, 
        dataset: DatasetBase
    ):
        self.dataset = dataset
        self.docs = self.dataset.gt_split_docs()

    def query(
        self,
        query: str,
        k: int = 5,
        verbose: bool = False,
    ) -> list[Document]:

        bm25_retriever = langchain_community.retrievers.BM25Retriever.from_documents(self.docs, k=k)
        res = bm25_retriever.get_relevant_documents(query)
        
        if verbose:
            for i, doc in enumerate(res):
                print(f"doc_rank {i+1}")
                print(f"doc {doc.page_content}")
                print()
                
        return res

class VectorRetriever(RetrieverBase):
    def __init__(
        self, 
        dataset: DatasetBase,
        model_name: str = "intfloat/e5-large-v2",
    ):
        self.dataset = dataset
        self.docs = self.dataset.gt_split_docs()
        self.model_name = model_name
        
        self.hf_emb = HuggingFaceEmbeddings(
            model_name="intfloat/e5-large-v2",
            model_kwargs={'device': 'cpu' if not torch.cuda.is_available() else 'cuda'},
        )
        print("making chroma db")
        self.db = Chroma.from_documents(self.docs, self.hf_emb)

    def query(
        self,
        query: str,
        k: int = 5,
        verbose: bool = False,
    ) -> list[Document]:

        res = self.db.similarity_search(query, k=k)
        
        if verbose:
            for i, doc in enumerate(res):
                print(f"doc_rank {i+1}")
                print(f"doc {doc.page_content}")
                print()
                
        return res


# modified from RAGatouilleLangChainRetriever
class RAGatouilleInMemoryLangChainRetriever(BaseRetriever):
    model: Any
    kwargs: dict = {}
    k: int

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Document]:
        """Get documents relevant to a query."""
        docs = self.model.search_encoded_docs(query, k=self.k)
        return [
            Document(
                page_content=doc["content"], metadata=doc.get("document_metadata", {})
            )
            for doc in docs
        ]

class RAGatouilleRetriever(RetrieverBase):
    def __init__(
        self, 
        dataset: DatasetBase,
        dataset_identifier: str,
        doc_maxlen: int = 1000,
        do_index: bool = False,
        in_memory: bool = True,
    ):
        self.dataset = dataset
        self.docs = self.dataset.gt_split_docs()
        self.doc_maxlen = doc_maxlen
        self.in_memory = in_memory
        
        index_name = f"RAGatouilleRetriever-{dataset_identifier}"
        print("INDEX NAME")
        print(index_name)
        
        if do_index:
            self.colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            self.colbert.index(
                collection=[d.page_content for d in self.docs],
                document_metadatas=[d.metadata for d in self.docs],
                index_name = index_name,
                max_document_length = doc_maxlen,
                split_documents = False, # requires docs to have been split into reasonable size
            )
        else:
            if in_memory:
                self.colbert = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
                self.colbert.encode(
                    [d.page_content for d in self.docs],
                    document_metadatas=[d.metadata for d in self.docs],
                    max_document_length = doc_maxlen,
                )
            else:
                self.colbert = RAGPretrainedModel.from_index(f".ragatouille/colbert/indexes/{index_name}")
            
    def query(
        self,
        query: str,
        k: int = 5,
        verbose: bool = False,
    ) -> list[Document]:

        if self.in_memory:
            langchain_retriever = RAGatouilleInMemoryLangChainRetriever(model=self.colbert, k=k)
            return langchain_retriever.invoke(query)
        else:
            langchain_retriever = self.colbert.as_langchain_retriever(k=k)
            return langchain_retriever.invoke(query)