import os

from typing import List

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FaissDB:
    def __init__(self, file_path: str, faiss_path: str, embeddings_model: Embeddings):
        self.text_docs = self.create_docs(file_path)
        self.faiss_db = self.create_faiss_db(faiss_path, self.text_docs, embeddings_model)

    @staticmethod
    def create_docs(
            file_path: str,
            separator: str = "\n\n",
            is_separator_regex: bool = False
    ) -> List[Document]:
        text_loader = TextLoader(file_path=file_path)
        text_docs = text_loader.load()
        text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=is_separator_regex
        )
        text_docs = text_splitter.split_documents(text_docs)
        return text_docs

    @staticmethod
    def create_faiss_db(
            faiss_path: str,
            text_docs: List[Document],
            embeddings_model: Embeddings
    ) -> FAISS:
        if os.path.exists(faiss_path) and os.listdir(faiss_path):
            files = os.listdir(faiss_path)
            matching_files = [file for file in files if file.endswith(".faiss")]
            if matching_files:
                return FAISS.load_local(faiss_path, embeddings=embeddings_model)
        faiss_db = FAISS.from_documents(documents=text_docs, embedding=embeddings_model)
        faiss_db.save_local(faiss_path)
        return faiss_db
