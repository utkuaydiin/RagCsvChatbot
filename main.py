
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. Open the CSV file in reading mode and the TXT file in writing mode
with open(r'C:\Users\asus\Desktop\Python\Rag\pythonProject\cleaned_web_traffic_data (1).csv', 'r') as f_in, open(r'C:\Users\asus\Desktop\Python\Rag\pythonProject\text.txt', 'w') as f_out:
    # 2. Read the CSV file and store in variable
    content = f_in.read()
    # 3. Write the content into the TXT file
    f_out.write(content)

llm = ChatOpenAI(model = "gpt-4")

loader = TextLoader(file_path=r"C:\Users\asus\Desktop\Python\Rag\pythonProject\text.txt")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
document = text_splitter.split_documents(data)

embeddings_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=document, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#daha önce yazılan rag promptu hubdan çeker
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in data)

rag_chain = (
    {"context":retriever | format_docs, "question":RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    print("Welcome! Ask any question about the data or type 'exit' to stop.")

    # Keep the interaction going until the user decides to exit
    while True:
        # Prompt the user for input
        user_input = input("You: ")

        # Exit condition: stop the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        # Stream the response in real-time as the model generates it
        print("AI: ", end="", flush=True)
        for chunk in rag_chain.stream(user_input):
            print(chunk, end="", flush=True)
        print()




