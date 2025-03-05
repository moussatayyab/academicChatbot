import streamlit as st
import os
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
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
text_data=get_pdf_text(files[:1])


# # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(text_data)
# print the number of chunks obtained
# len(text_chunks)



modelPath = "BAAI/bge-large-en-v1.5"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}
#if using apple m1/m2 -> use device : mps (this will use apple metal)

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': True}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)

# Convert text_chunks (list of strings) to Document objects
documents = [Document(page_content=chunk) for chunk in text_chunks]  

# Now use 'documents' instead of 'text_chunks'
vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vector_store.as_retriever()


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


prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


st.title("Student Assistant")

selections=st.sidebar.selectbox("☰ Menu", ["Home","AI Assistant", "Feedback")


query=""
query=st.text_input("Write Query Here")


if selections=="Home":
    st.markdown("""### **About the School Student Assistant Chatbot**  

    The **School Student Assistant Chatbot** is an AI-powered virtual assistant designed to help students with their academic and school-related queries. It provides instant responses to common questions, assists with homework, shares important school updates, and offers guidance on schedules, subjects, and extracurricular activities.  
    
    This chatbot enhances student engagement by providing **quick, accurate, and interactive support** anytime, reducing the need for manual inquiries. It is built with advanced **natural language processing (NLP)** to understand student queries effectively and deliver relevant information in a conversational manner.  
    
    ### **Key Features:**  
    ✅ **Homework Assistance** – Provides explanations and study resources.  
    ✅ **Timetable & Schedule Support** – Helps students check class schedules.  
    ✅ **School Announcements & Notices** – Delivers updates on events and policies.  
    ✅ **Subject Guidance** – Answers subject-related queries.  
    ✅ **Interactive & Voice Support** – Allows students to communicate via text or voice.  
    
    With its **user-friendly interface** and **AI-driven capabilities**, the chatbot enhances the learning experience by making information more accessible and helping students stay organized. """)


if st.button("Submit") and query!="":
    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_openai
    | StrOutputParser()
    )
    st.subheader("OpenAI GPT Response")
    res=rag_chain.invoke(query)
    st.write(res)

    # # performing a similarity search to fetch the most relevant context
    st.write("")
    st.write("")
    st.write("")

    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm_llama3
    | StrOutputParser()
    )
    st.subheader("Meta Llama3 GPT Response")
    res=rag_chain.invoke(query)
    st.write(res)

    # context=""

    # for i in vector_store.similarity_search(query):
    #     context += i.page_content 

    # Get current date and time
    # now = datetime.now()
    # # Format as string
    # formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    # pd.DataFrame({"DateTime":formatted_now,"Context":context,"AI Response":res,"User Feedback":""}.to_excel("user_feedback")
                 





