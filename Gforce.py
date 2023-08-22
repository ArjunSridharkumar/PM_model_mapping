import streamlit as st
import datetime
import os
from os import environ
import PyPDF2
from langchain.vectorstores import DeepLake
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.agents.agent_toolkits import ZapierToolkit
# from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pysqlite3
from langchain.chat_models import ChatOpenAI
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import qdrant_client
from qdrant_client import QdrantClient,models
from qdrant_client.http.models import PointStruct
from langchain.agents import initialize_agent
from langchain.vectorstores import Qdrant
# from zap import schedule_interview
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter


openai_api_key = st.secrets["OPENAI_API_KEY"]
deeplake_key = st.secrets["ACTIVELOOP_TOKEN"]
# QDRANT_COLLECTION ="resume"



def generate_response(doc_texts, openai_api_key, query_text):
    doc_texts = None
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7,openai_api_key=openai_api_key)
    
    # Split documents into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.create_documents(doc_texts)
    
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    custom_prompt_template = """You are project planner. You will be given a codebase and will have to break it down into subtasks for teams to develop/
    Plan out out each task and subtask step by step. Plan tasks only relevant to the provided document. Do not make up irrelevant tasks./
    Be helpful and answer in detail while preferring to use information from provided documents.
    Task: Prepare  in 3 paragraphs
    Topic: Project Planning
    Style: Technical
    Tone: Professional
    Audience: Project Manager

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])


    
    db = DeepLake(dataset_path="hub://arjunsridhar9720/twitter_clone_org", token = deeplake_key,read_only=True, embedding_function=embeddings)
    retriever = db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['fetch_k'] = 100
    retriever.search_kwargs['maximal_marginal_relevance'] = True
    retriever.search_kwargs['k'] = 10
    model = ChatOpenAI(model='gpt-4') # switch to 'gpt-4'
    qa = ConversationalRetrievalChain.from_llm(model,retriever=retriever)
    # chain_type_kwargs = {"prompt": PROMPT}
    qa =  RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=retriever,
                                       return_source_documents=False,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    response = qa({'query': query_text})
    print (response)
    return response["result"]
    
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "You are a Q&A chatbot that answers questions based on uploaded files"}]

# Page title
st.set_page_config(page_title='Gforce Resume Assistant', layout='wide')
st.title('Gforce Resume Assistant')

# File upload
# uploaded_files = st.file_uploader('Please upload you resume(s)', type=['pdf','txt'], accept_multiple_files=True)

# Query text
query_text = st.text_input('Enter your question:', placeholder='Select candidates based on experience and skills')

# Initialize chat placeholder as an empty list
if "chat_placeholder" not in st.session_state.keys():
    st.session_state.chat_placeholder = []

# Form input and query
# if st.button('Submit', key='submit_button'):
uploaded_files = True
if openai_api_key.startswith('sk-'):
    if uploaded_files and query_text:
        # documents = [read_pdf_text(file) for file in uploaded_files]
        # documents = uploaded_files
        with st.spinner('Chatbot is typing...'):
            documents = None
            response = generate_response(documents,openai_api_key, query_text)
            st.session_state.chat_placeholder.append({"role": "user", "content": query_text})
            st.session_state.chat_placeholder.append({"role": "assistant", "content": response})

        # Update chat display
        for message in st.session_state.chat_placeholder:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    else:
        st.warning("Please upload one or more PDF files and enter a question to start the conversation.")

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.session_state.chat_placeholder = []
    uploaded_files.clear()
    query_text = ""
    st.empty()  # Clear the chat display

st.button('Clear Chat History', on_click=clear_chat_history)

# Create a sidebar with text input boxes and a button
# st.sidebar.header("Schedule Interview")
# person_name = st.sidebar.text_input("Enter Person's Name", "")
# person_email = st.sidebar.text_input("Enter Person's Email Address", "")
# date = st.sidebar.date_input("Select Date for Interview")
# time = st.sidebar.time_input("Select Time for Interview")
# schedule_button = st.sidebar.button("Schedule Interview")

# if schedule_button:
#     if not person_name:
#         st.sidebar.error("Please enter the person's name.")
#     elif not person_email:
#         st.sidebar.error("Please enter the person's email address.")
#     elif not date:
#         st.sidebar.error("Please select the date for the interview.")
#     elif not time:
#         st.sidebar.error("Please select the time for the interview.")
#     else:
#         # Call the schedule_interview function from the zap.py file
#         success = schedule_interview(person_name, person_email, date, time)

#         if success:
#             st.sidebar.success("Interview Scheduled Successfully!")
#         else:
#             st.sidebar.error("Failed to Schedule Interview")
