import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == '__main__':
    print("Retrieving...")
    query = "What is pinecone in machine learning"
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    llm = ChatVertexAI(model_name="gemini-2.5-flash")

    vectorstore = PineconeVectorStore.from_existing_index(os.environ["INDEX_NAME"], embeddings)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)
    result = retrieval_chain.invoke(input={"input":query})
    print(result)

    template = """
    Use the following pieces of context to answer the context at the end.
    If you do not the answer, just say that you do not know the answer, don't try to make up the answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "Thanks for asking!" at the end of the answer.
    {context}
    
    question: {question}
    Helpful answer:
    """

    custom_rag_template = PromptTemplate.from_template(template)
    rag_chain = (
        {"context":vectorstore.as_retriever() | format_docs, "question":RunnablePassthrough()}
        | custom_rag_template
        | llm
    )

    rag = rag_chain.invoke(query)
    print(rag)