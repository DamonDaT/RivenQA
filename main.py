import gradio
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.base import Chain

from src.chains import RivenQA
from src.vdbs import FaissDB

# Initialize environment variables from .env file
load_dotenv(verbose=True)

# Global
Riven_QA: Chain


def initialize_riven_qa(file_path: str, faiss_path: str) -> Chain:
    # Model
    embeddings_model = OpenAIEmbeddings()
    llm = ChatOpenAI()

    # Vector DB
    faiss_db = FaissDB(file_path=file_path, faiss_path=faiss_path, embeddings_model=embeddings_model).faiss_db

    # QA Chain
    global Riven_QA
    Riven_QA = RivenQA(llm=llm, vector_store=faiss_db).chain
    Riven_QA.return_source_documents = True

    return Riven_QA


def chat_bot(message) -> str:
    ans = Riven_QA.invoke({"query": message})

    if ans["source_documents"]:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]

    else:
        return "Sorry, there is no answer to this question in the knowledge base."


def launch_gradio():
    demo = gradio.ChatInterface(
        fn=chat_bot,
        title="RivenQA",
        chatbot=gradio.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")


if __name__ == "__main__":
    initialize_riven_qa(file_path="", faiss_path="")
    launch_gradio()
