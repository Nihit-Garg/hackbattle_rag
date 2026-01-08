from langchain_chroma import Chroma

from langchain_core.prompts import PromptTemplate

from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


#text file ingestion 
loader = TextLoader(
    "IEEE-CS Rag Task/hackbattle_doc.txt",
    encoding="utf-8"
)

docs = loader.load()
script = docs[0].page_content


#chunking 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

chunks = splitter.create_documents([script])


#vector store and embeddings 
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model
)


retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)



llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.2
)

#implementing HyDe
hyde_llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0.0 
)

#hyde prompt
hyde_prompt = PromptTemplate(
    template="""
You are writing an official informational paragraph for Hackbattle,
a hackathon conducted at VIT Vellore.
Based on the question below, write a concise and factual paragraph
that could appear in an official Hackbattle information document.
Do not add assumptions or new facts.

Question:
{question}""",
    input_variables=["question"]
)


#prompt template
prompt = PromptTemplate(
    template="""
You are a helpful assistant answering questions about Hackbattle.

Use only the information given to you.
Do not add assumptions or external knowledge.

If the answer is not known, respond with "I don't know." and stop.

Do NOT mention the words "context", "provided context", "document", "file", or "source" in your response.

Conversation history:
{chat_history}

Context:
{context}

Question:
{question}
""",
    input_variables=["chat_history", "context", "question"]
)

#helper functions 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    return "\n".join(history)


parser = StrOutputParser()

#hyde in chain 
hyde_chain = (
    hyde_prompt| hyde_llm| parser
)

parallel_chain = RunnableParallel({
    "context": hyde_chain | retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

lambda_chain = RunnableLambda(lambda x: {
    "context": x["context"],
    "question": x["question"],
    "chat_history": format_history(chat_history)
})

final_chain = parallel_chain | lambda_chain | prompt | llm | parser


#looping container 
chat_history = []

print("\nAsk your query regarding HACKBATTLE (type 'exit' to quit)\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("\n Exiting chatbot. Bye!\n")
        break

    answer = final_chain.invoke(question)

    print(f"\nAssistant: {answer}\n")

    # store memory
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {answer}")

    # limit memory size
    if len(chat_history) > 12:
        chat_history = chat_history[-12:]
