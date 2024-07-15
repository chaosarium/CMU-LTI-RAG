#! Read and preprocess text data
# 
# - TextLoader loads txt files 
# - TextLoader tries to load pdf files (if we need to use it)
# 
# We could in the futere read markdown or HTML files by importing more stuff from langchain_community

from re import split
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TextSplitter
import warnings
from pprint import pprint
warnings.filterwarnings(action="ignore", message=".*a chunk of size.*")

class DatasetBase:
    def __init__(self,
        text_splitter: TextSplitter,
    ) -> None:
        self.docs: list[Document]
        self.split_docs: list[Document]
        self.doc_max_len: int
        self.split_doc_max_len: int
        self.text_splitter = text_splitter
        
    def gt_docs(self) -> list[Document]:
        return self.docs
    def gt_split_docs(self) -> list[Document]:
        return self.split_docs
    def gt_doc_max_len(self) -> int:
        return max(map(lambda d: len(d.page_content), self.docs))
    def gt_split_doc_max_len(self) -> int:
        return max(map(lambda d: len(d.page_content), self.split_docs))
    def __getitem__(self, i: int) -> Document:
        return self.split_docs[i]
    def print_summary(self) -> None:
        print(f"this set contains {len(self.docs)} documents with max len {self.gt_doc_max_len()}")
        print(f"split into {len(self.split_docs)} documents with max len {self.gt_split_doc_max_len()}")

splitter_choices = {
    'char_text_splitter': CharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=250,
        length_function=len,
        is_separator_regex=False,
    ),
    'recursive_char_text_splitter': RecursiveCharacterTextSplitter(
        chunk_size=650,
        chunk_overlap=250,
        length_function=len,
        is_separator_regex=False,
    )
}

class TextDataset(DatasetBase):
    def __init__(
        self,
        data_dir: str,
        text_splitter: TextSplitter,
        metadata_augment: bool = False, # whether to prepend metadata to text content
    ) -> None:
        super().__init__(text_splitter)
        
        loader = DirectoryLoader(data_dir, glob="**/*.txt", show_progress=True, use_multithreading=True, loader_cls=TextLoader)
        self.docs = loader.load()
        self.prepend_metadata = metadata_augment
        
        split_docs = self.text_splitter.split_documents(self.docs)
        if metadata_augment:
            for doc in split_docs:
                doc.page_content = f"[Excerpt from {os.path.splitext(os.path.basename(doc.metadata['source']))[0]}]\n{doc.page_content}"

        self.split_docs = split_docs

class PdfDataset(DatasetBase):
    def __init__(
        self,
        data_dir: str,
        text_splitter: TextSplitter,
    ) -> None:
        super().__init__(text_splitter)
        
        # loader = DirectoryLoader(data_dir, glob="**/*.pdf", show_progress=True, use_multithreading=True, loader_cls=PDFMinerLoader)
        loader = DirectoryLoader(data_dir, glob="**/*.pdf", show_progress=True, use_multithreading=True, loader_cls=PyPDFLoader)
        self.docs = loader.load()
        self.split_docs = self.text_splitter.split_documents(self.docs)

class ComposedDataset(DatasetBase):
    def __init__(
        self,
        A: DatasetBase,
        B: DatasetBase,
    ) -> None:
        self.text_splitter = None
        self.A, self.B = A, B
    
    def gt_docs(self) -> list[Document]:
        return self.A.gt_docs() + self.B.gt_docs()
    def gt_split_docs(self) -> list[Document]:
        return self.A.gt_split_docs() + self.B.gt_split_docs()
    def gt_doc_max_len(self) -> int:
        return max(self.A.gt_doc_max_len(), self.B.gt_doc_max_len())
    def gt_split_doc_max_len(self) -> int:
        return max(self.A.gt_split_doc_max_len(), self.B.gt_split_doc_max_len())
    def __getitem__(self, i: int) -> Document:
        return self.gt_split_docs()[i]
    def print_summary(self) -> None:
        print("Component A:")
        self.A.print_summary()
        print("Component B:")
        self.B.print_summary()