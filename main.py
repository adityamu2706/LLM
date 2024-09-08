import os
import streamlit as st
import pickle
import time
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import retrieval_qa
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


st.title("LLM for Restaurant Reccomendation")
st.sidebar.title("Links to restaurants")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
os.environ['OPENAI_API_KEY'] = "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"
llm = OpenAI(temperature=0.9)

embeddings = 0
pkl = 0
db = None
if process_url_clicked:
    # load data
    loader = AsyncHtmlLoader(urls)
    data = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(data[0:500])
    #print(docs_transformed[0].page_content[0:500])

    main_placeholder.text("Data Loading...Started...✅✅✅")

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=300
    )

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(data, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pkl = db.serialize_to_bytes()  # serializes the faiss
        pickle.dump(pkl, f)

query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=pkl)
            memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=False)
            chain = ConversationalRetrievalChain.from_llm(llm= ChatOpenAI(temperature=0.2, model_name='gpt-3.5-turbo'),chain_type="stuff",retriever=db.as_retriever(),
    get_chat_history=lambda o:o,
    memory = memory,
    return_generated_question=True,
    verbose=False
)
            
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
                