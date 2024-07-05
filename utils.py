import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import os
import requests
from googlesearch import search



load_dotenv(f"{Path.cwd()}/config.env")
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

model = genai.GenerativeModel('gemini-1.5-pro')
model_flash = genai.GenerativeModel('gemini-1.5-flash')

def internet_context(search_term):
    context = {}
    url_list = search(term=search_term, num_results=10)
    for url in url_list:
        print(url)
        html_content = requests.get(url=url)
        print(html_content.text)

def send_message(user_input,_):
    context = internet_context(user_input["text"])
    response = model.generate_content(user_input["text"])
    return response.text


internet_context("ai chatbots")