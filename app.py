from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

from datasets import load_dataset

import cassio
import streamlit as st
from PyPDF2 import PdfReader

ASTRA_DB_APPLICATION_TOKEN = "AstraCS:UPkfqhgxqlGClRZQaoNRZTIP:22e71b1cb4a916d3722697a89237aed24cc6b872b72bad42ee11d8c26133710e"
ASTRA_DB_ID = "4e301076-f4ed-46a6-af16-1ae99fc5b780"
OPENAI_API_KEY = "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"

pdfreader = PdfReader("budget_speech.pdf")

from typing_extensions import Concatenate

raw_text = ""

for i, page in enumerate(pdfreader.pages):
  content = page.extract_text()
  if content:
    raw_text += content


cassio.init(token = ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)


llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.6)
embedding = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Function to load OpenAI model and get response
def get_openAI_respnse(question):
    llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
    response = llm(question)
    return response


astra_vector_store = Cassandra(
    embedding=embedding,
    table_name = "mini_qa_demo",
    session = None,
    keyspace = None
)


from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size = 800,
    chunk_overlap = 200,
    length_function = len
)

texts = text_splitter.split_text(raw_text)


astra_vector_store.add_texts(texts)
astra_vextor_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)




## Intitialize Streamlit app
st.set_page_config(page_title = "Ask Questions from the India Budget 2023 PDF")
st.header("PDF_QA")

input = st.text_input("Enter your question here", key="input").strip()
response = astra_vextor_index.query(input, llm=llm)

submit = st.button("Generate")


#If submit button is clicked
if submit:
    st.subheader("The response is")
    st.write(response)
