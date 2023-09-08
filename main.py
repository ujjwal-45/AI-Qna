import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# function to extract pdf pages return list of page texts

def read_and_textify(files):
  text_list = []
  sources_list = []
  for file in files:
    pdfReader = PyPDF2.PdfReader(file)

    for i in range(len(pdfReader.pages)):
      pageObj = pdfReader.pages[i]
      text = pageObj.extract_text()
      pageObj.clear()
      text_list.append(text)
      sources_list.append(file.name + "_page_" + str(i))
  return [text_list, sources_list]

st.set_page_config(layout="centered", page_title="AI_Qna")
st.header("AI_Qna")
st.write("---")

# for uploading files

uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])
st.write("---")

if uploaded_files is None:
  st.info(f"""Upload files to analyze""")
elif uploaded_files:
  st.write(str(len(uploaded_files)) + "document(s) loaded...")

  textify_output = read_and_textify(uploaded_files)

  documents = textify_output[0]
  sources = textify_output[1]

  # extraxt the embeddings
  embeddings = OpenAIEmbeddings(openai_api_key = st.secrets["openai_api_key"])
  
  # storing page numbers in vector store
  vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])
  
  model_name = "gpt-3.5-turbo"
  
  retriever = vStore.as_retriever()
  retriever.search_kwargs = {'k': 2}
  
  llm = OpenAI(model_name=model_name, openai_api_key = st.secrets["openai_api_key"], streaming=True)
  model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
  
  st.header("Ask your questions")
  user_q = st.text_area("Enter your questions here")

  if st.button("Get Response"):
    try:
      with st.spinner("Model is working on it..."):
        result = model({"question": user_q}, return_only_outputs=True)
        st.subheader('Your response:')
        st.write(result['answer'])
        st.subheader('Source pages:')
        st.write(result['sources'])
    except Exception as e:
      st.error(f"An error occurred: {e}")
      st.error('Oh no, the GPT response resulted in an error : (Please try again with a different question.')






