import streamlit as st
import getpass
import os


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from pytube import YouTube


from langchain_groq import ChatGroq


from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
#os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
groq_api_key=st.secrets("GROQ_API_KEY")

#Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful massistant . Please  repsonse to the user queries"),
        ("user","Question:{question}")
    ]
)


## Gemma Model USsing Groq API
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

#streamlit app
st.set_page_config(page_title="Langchain: Ask Anything from Google Gemma",page_icon="🦜")
st.title("🦜Using Langchain: Ask Anything from Google Gemma")

footer = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        color: #555;
    }
    </style>
    <div class="footer">
        © 2025 Created by Manish Vivek
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)

#Main Interface for the user
st.write("Please go ahead and write your question")
user_input=st.text_input("You:")

with st.spinner("Waiting---"):
    if user_input:
        output_parser=StrOutputParser()
        chain=prompt|llm|output_parser
        answer=chain.invoke({'question':user_input})
        st.write(answer)
    



