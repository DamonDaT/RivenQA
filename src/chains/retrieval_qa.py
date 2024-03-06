from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore


class RivenQA:
    def __init__(self, llm: BaseLanguageModel, vector_store: VectorStore):
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.8}
            )
        )
