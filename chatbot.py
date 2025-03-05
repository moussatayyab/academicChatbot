import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import getpass
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
import gdown
###############################################################################################
files=os.listdir()
###############################################################################################
files=[file for file in files if file.split(".")[-1]=="pdf"]

###############################################################################################
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text
# text_data=get_pdf_text(files[:1])


# # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# text_chunks = text_splitter.split_text(text_data)
# # print the number of chunks obtained
# # len(text_chunks)



# modelPath = "BAAI/bge-large-en-v1.5"

# # Create a dictionary with model configuration options, specifying to use the CPU for computations
# model_kwargs = {'device':'cpu'}
# #if using apple m1/m2 -> use device : mps (this will use apple metal)

# # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
# encode_kwargs = {'normalize_embeddings': True}

# # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
# embeddings = HuggingFaceEmbeddings(
#     model_name=modelPath,     # Provide the pre-trained model's path
#     model_kwargs=model_kwargs, # Pass the model configuration options
#     encode_kwargs=encode_kwargs # Pass the encoding options
# )

# # Convert text_chunks (list of strings) to Document objects
# documents = [Document(page_content=chunk) for chunk in text_chunks]  

# # Now use 'documents' instead of 'text_chunks'
# vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vector_store.as_retriever()


#############################################################################################################################
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm_llama3 = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

################################################################################################################################


################################################################################################################################
file_id = "1ug8pf1M1tes-CJMhS_sso372tvC4RQv8"
output_file = "open_ai_key.txt"

def download_db():
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
    return output_file
k=""
with open(download_db(),'r') as f:
    f=f.read()
    # st.write(f)
    k=f
    

os.environ["OPENAI_API_KEY"] = k
llm_openai = ChatOpenAI(model="gpt-4o-mini")
######################################################################################################################################


# prompt = hub.pull("rlm/rag-prompt")


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


st.title("Student Assistant")

selections=st.sidebar.selectbox("☰ Menu", ["Home","AI Assistant", "Feedback"])


query=""


if selections=="Home":
    st.markdown("""The School Student Assistant Chatbot is an AI-powered virtual assistant designed to help students with their academic and school-related queries. It provides instant responses to common questions, assists with homework, shares important school updates, and offers guidance on schedules, subjects, and extracurricular activities.  

    
     Key Features:  
    ✅ Homework Assistance – Provides explanations and study resources.  
    ✅ Timetable & Schedule Support – Helps students check class schedules.  
    ✅ School Announcements & Notices – Delivers updates on events and policies.  
    ✅ Subject Guidance – Answers subject-related queries.  
    ✅ Interactive & Voice Support – Allows students to communicate via text or voice.  """)
    
    

if selections=="AI Assistant":
    query=st.text_input("Write Query Here")
    # if st.button("Submit") and query!="":
    #     rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm_openai
    #     | StrOutputParser()
    #     )
    #     st.subheader("OpenAI GPT Response")
    #     res=rag_chain.invoke(query)
    #     st.write(res)
    
    #     # # performing a similarity search to fetch the most relevant context
    #     st.write("")
    #     st.write("")
    #     st.write("")
    
    #     rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm_llama3
    #     | StrOutputParser()
    #     )
    #     st.subheader("Meta Llama3 GPT Response")
    #     res=rag_chain.invoke(query)
    #     st.write(res)

    # context=""

    # for i in vector_store.similarity_search(query):
    #     context += i.page_content 

    # Get current date and time
    # now = datetime.now()
    # # Format as string
    # formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    # pd.DataFrame({"DateTime":formatted_now,"Context":context,"AI Response":res,"User Feedback":""}.to_excel("user_feedback")
                 
# https://docs.google.com/spreadsheets/d/1ramLbRPuTHo4yY2ylTUIM1vyy539MumNGAdFCP5w9uY/edit?usp=sharing
# https://docs.google.com/spreadsheets/d/1k1MYDZ7n9sIjPTfXFMFHMwEJOMkmhcWzikFVoXlH2SQ/edit?usp=sharing

if selections=="Feedback":
    
    st.subheader("Welcome to User Feedback Section")
    
    st.write("Please Leave Feedback [Here](https://docs.google.com/forms/d/e/1FAIpQLSekxnpLx5glG_bYHy54m0IrbBIZxEM37dihnBNOeRMR0n9KUg/viewform?usp=header)")
    sheet_id = '1k1MYDZ7n9sIjPTfXFMFHMwEJOMkmhcWzikFVoXlH2SQ' # replace with your sheet's ID
    
    url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df=pd.read_csv(url)
    st.write(df.columns)

    

    col1,col2=st.columns(2)
    with col1:
        ratings_x=df['How satisfied are you with the chatbot\'s overall performance?'].value_counts().index
        ratings_y=df['How satisfied are you with the chatbot\'s overall performance?'].value_counts().values
        st.subheader("Application Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=ratings_x, y=ratings_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Ratings")
        ax.set_ylabel("Values")
        ax.set_title("Application Ratings")
        st.pyplot(fig)

    
    with col2:
        effective_resources_x=df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().index
        effective_resources_y=df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().values
        st.subheader("Application Effective")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=effective_resources_x, y=effective_resources_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Application Effective")
        st.pyplot(fig)
        
        
        
    col1,col2=st.columns(2)
    with col1:
        response_x=df['Which GPT responses do you find the most helpful?'].value_counts().index
        response_y=df['Which GPT responses do you find the most helpful?'].value_counts().values
        st.subheader("Application Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=response_x, y=response_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("GPT Response ")
        st.pyplot(fig)

    
    with col2:
        interaction_x=df['How easy was it to interact with the chatbot?'].value_counts().index
        interaction_y=df['How easy was it to interact with the chatbot?'].value_counts().values
        st.subheader("Application Effective")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=interaction_x, y=interaction_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Application Effective")
        st.pyplot(fig)

        
        
        
        
    col1,col2=st.columns(2)
    with col1:
        satisfactory_x=df['Was the chatbot\'s response time satisfactory?'].value_counts().index
        satisfactory_y=df['Was the chatbot\'s response time satisfactory?'].value_counts().values
        st.subheader("Application Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=satisfactory_x, y=satisfactory_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("GPT Response ")
        st.pyplot(fig)

    
    with col2:
        understand_x=df['Did the chatbot understand your questions correctly?'].value_counts().index
        understand_y=df['Did the chatbot understand your questions correctly?'].value_counts().values
        st.subheader("Application Effective")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=understand_x, y=understand_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Application Effective")
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        st.write(df.columns[8])
        response_x=df[df.columns[8]].value_counts().index
        response_y=df[df.columns[8]].value_counts().values
        st.subheader("Application Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=satisfactory_x, y=satisfactory_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("GPT Response ")
        st.pyplot(fig)

    
    with col2:
        experience_x=df['Up to what extent this chatbot contributed to your learning experience or academic efficiency?'].value_counts().index
        experience_y=df['Up to what extent this chatbot contributed to your learning experience or academic efficiency?'].value_counts().values
        st.subheader("Application Effective")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=experience_x, y=experience_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Application Effective")
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        relevant_x=df[df.columns[8]].value_counts().index
        relevant_y=df[df.columns[8]].value_counts().values
        st.subheader("Application Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=relevant_x, y=relevant_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("GPT Response ")
        st.pyplot(fig)

    
    with col2:
        recommend_x=df['Would you recommend this chatbot to fellow students or faculty members for academic support?'].value_counts().index
        recommend_y=df['Would you recommend this chatbot to fellow students or faculty members for academic support?'].value_counts().values
        st.subheader("Application Effective")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=recommend_x, y=recommend_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Application Effective")
        st.pyplot(fig)
        
        
        
        
        
        
        # st.subheader("Pie Chart (Matplotlib)")
        # fig, ax = plt.subplots(figsize=(4,2))
        # ax.pie(df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().values, labels=df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),radius=0.6, textprops={'fontsize': 8})
        # # Remove extra padding
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # ax.set_title("Pie Chart Example")
        # # Adjust layout to reduce whitespace
        # st.pyplot(fig)


    

