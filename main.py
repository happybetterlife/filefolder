import os

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_CALLBACKS_MANAGER"] = "false"

__import__('pysqlite3')  # Ensure sqlite3 is imported to avoid errors with langchain
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import pandas as pd
from io import StringIO
from dotenv import load_dotenv
load_dotenv()

#Title
st.title("File Folder")
st.write("---")

#File Uploade
uploaded_file = st.file_uploader("파일을 업로드하세요", type=["pdf", "txt", "csv"])
st.write("---")

def file_to_documents(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_filepath)
        pages = loader.load_and_split()

    elif file_extension == ".txt":
        loader = TextLoader(temp_filepath, encoding='utf-8')
        pages = loader.load_and_split()

    elif file_extension == ".csv":
        loader = CSVLoader(file_path=temp_filepath,
            encoding='utf-8',  # 인코딩 지정
            csv_args={
                'delimiter': ',',  # 구분자
                'quotechar': '"',  # 인용 문자
            }
        )
        pages = loader.load()  # CSV는 보통 split 하지 않음

    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    temp_dir.cleanup()
    return pages

#Loader

if uploaded_file is not None:
    pages = file_to_documents(uploaded_file)
    st.write(f"파일 '{uploaded_file.name}'이(가) 성공적으로 업로드되었습니다.")
    st.write(f"문서의 페이지 수: {len(pages)}")

#Splitter
text_splitter = RecursiveCharacterTextSplitter(
    # Set a reallysmall chunk size, just to show.
    chunk_size=300, 
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(pages)


#Embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # With the 'text-embedding-3-large',
    # of models, you can specify the size
    # of the embeddings you want to return.
    # dimensions=1024
)

import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

#Chroma DB
db = Chroma.from_documents(
    texts,
    embeddings_model)

#User Input
st.header("질문을 입력하세요")
question = st.text_input("질문을 입력하세요", placeholder="예: 이 논문의 주요 기여는 무엇인가요?")

if st.button("질문하기"):
    with st.spinner("답변을 생성하는 중..."):
        #Retriever
        llm = ChatOpenAI(temperature=0)
        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=db.as_retriever(), llm=llm
        )

        #Prompt Template
        prompt = hub.pull("rlm/rag-prompt")

        #Generate
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        rag_chain = (
            {"context": retriever_from_llm | format_docs, 
            "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        #Question
        result = rag_chain.invoke("What is the main contribution of this paper?")
        st.write(result)
