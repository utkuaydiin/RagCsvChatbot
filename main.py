
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from typing import Any, Iterator, List, Union
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

load_dotenv()

# 1. Open the CSV file in reading mode and the TXT file in writing mode
with open(r'C:\Users\asus\Desktop\Python\Rag\pythonProject\cleaned_web_traffic_data (1).csv', 'r') as f_in, open(r'C:\Users\asus\Desktop\Python\Rag\pythonProject\text.txt', 'w') as f_out:
    # 2. Read the CSV file and store in variable
    content = f_in.read()
    # 3. Write the content into the TXT file
    f_out.write(content)

df = pd.read_csv(r"C:\Users\asus\Desktop\Python\Rag\pythonProject\cleaned_web_traffic_data (1).csv")
llm = ChatOpenAI(model = "gpt-4")


class BaseDataFrameLoader(BaseLoader):
    def __init__(self, data_frame: Any, *, page_content_column: Union[str, List[str]] = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: DataFrame object.
            page_content_column: Name of the column or list of column names containing the page content.
              Defaults to "text".
        """
        self.data_frame = data_frame
        self.page_content_column = page_content_column

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load records from dataframe."""

        for _, row in self.data_frame.iterrows():
            if isinstance(self.page_content_column, list):
                text = ' '.join(f'{col}:{row[col]}' for col in self.page_content_column)
            else:
                text = f'{col}:{row[self.page_content_column]}'
            metadata = row.to_dict()
            if isinstance(self.page_content_column, list):
                for col in self.page_content_column:
                    metadata.pop(col, None)
            else:
                metadata.pop(self.page_content_column, None)
            yield Document(page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """Load full dataframe."""
        return list(self.lazy_load())


class DataFrameLoader(BaseDataFrameLoader):
    """Load `Pandas` DataFrame."""

    def __init__(self, data_frame: Any, page_content_column: Union[str, List[str]] = "text"):
        """Initialize with dataframe object.

        Args:
            data_frame: Pandas DataFrame object.
            page_content_column: Name of the column or list of column names containing the page content.
              Defaults to "text".
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError(
                "Unable to import pandas, please install with `pip install pandas`."
            ) from e

        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError(
                f"Expected data_frame to be a pd.DataFrame, got {type(data_frame)}"
            )
        super().__init__(data_frame, page_content_column=page_content_column)


loader = DataFrameLoader(df, page_content_column=[
    "URL",
    "IP Address",
    "Timestamp",
    "Method",
    "URL",
    "Status",
    "Size",
    "User Agent"])

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




