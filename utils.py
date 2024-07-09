import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from googlesearch import search
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By




load_dotenv(f"{Path.cwd()}/config.env")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

with open(f"{Path.cwd()}/prompts/search_term.txt",'r') as prompt:
    search_prompt = prompt.read()

with open(f"{Path.cwd()}/prompts/conditional_agent_prompt.txt",'r') as main_agent_prompt:
    conditional_agent_prompt = main_agent_prompt.read()

with open(f"{Path.cwd()}/prompts/code_prompt.txt",'r') as code_prompt:
    claude_code_prompt = code_prompt

model = genai.GenerativeModel('gemini-1.5-pro')
model_flash = genai.GenerativeModel('gemini-1.5-flash')
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=os.environ['GEMINI_API_KEY'])
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100,length_function=len)
driver = uc.Chrome(headless=True)

def internet_context(user_query): 
    context = {}
    search_query_gen_prompt = PromptTemplate.from_template(template=search_prompt)
    formatted_search_query_prompt = search_query_gen_prompt.format(question=user_query)
    search_term = model_flash.generate_content(formatted_search_query_prompt,generation_config={"response_mime_type":"application/json"})
    url_list = list(search(term=json.loads(search_term.text)["search_query"], num_results=4))
    for url_index,url in enumerate(url_list):
        driver.get(url)
        parsed_url_text = driver.find_element(By.TAG_NAME,'body').text
        text_splits = text_splitter.split_text(parsed_url_text)
        db = FAISS.from_texts(text_splits, embedding=embedding_model)
        relevant_split = db.similarity_search(user_query,k=2)
        context[f"internet_context_{url_index + 1}"] = {"context" : ''.join(split.page_content for split in relevant_split),"url" : url}
    return context
    

def send_message(user_input,_):
    generated_message = model.generate_content(user_input["text"])
    return generated_message.text

test = internet_context("how to install windows on any pc ?")
print(test)