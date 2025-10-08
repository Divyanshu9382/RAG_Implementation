import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()



if __name__ == "__main__":
    print("Ingesting...")
    loader = TextLoader("D:\\Projects\\vectordbs\\mediumblog1.txt", encoding="utf-8")
    document = loader.load()
    print("Splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = text_splitter.split_documents(document)
    print("created {} chunks.".format(len(text)))

    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")

    print("Ingesting...")
    PineconeVectorStore.from_documents(text, embeddings, index_name=os.environ["INDEX_NAME"])
    print("Finished")