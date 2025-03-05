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


if selections=="Feedback":
    st.subheader("Welcome to User Feedback Section")
    
    st.write("Please Leave Feedback [Here](https://docs.google.com/forms/d/e/1FAIpQLSekxnpLx5glG_bYHy54m0IrbBIZxEM37dihnBNOeRMR0n9KUg/viewform?usp=header)")
    sheet_id = '1ramLbRPuTHo4yY2ylTUIM1vyy539MumNGAdFCP5w9uY' # replace with your sheet's ID
    
    url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df=pd.read_csv(url)
    st.write(df)


import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
data = {'Category': ['A', 'B', 'C', 'D'],
        'Values': [30, 20, 50, 40]}
df = pd.DataFrame(data)

st.title("Bar Plot and Pie Chart in Streamlit")

# Select visualization
chart_type = st.selectbox("Select Chart Type", ["Bar Plot", "Pie Chart"])

# Bar Plot using Seaborn
if chart_type == "Bar Plot":
    st.subheader("Bar Plot (Seaborn + Matplotlib)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x='Category', y='Values', data=df, palette="viridis", ax=ax)
    ax.set_xlabel("Category")
    ax.set_ylabel("Values")
    ax.set_title("Bar Plot Example")
    st.pyplot(fig)

# Pie Chart using Matplotlib
elif chart_type == "Pie Chart":
    st.subheader("Pie Chart (Matplotlib)")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(df['Values'], labels=df['Category'], autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    ax.set_title("Pie Chart Example")
    st.pyplot(fig)


    

