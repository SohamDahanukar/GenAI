from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os
import requests
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("UGGINGFACE_API_TOKEN")

def img2txt(url):
    # Set max_new_tokens to control the length of the generated text
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=50)
    
    # Get the generated text from the first dictionary in the list
    result = image_to_text(url)
    text = result[0]["generated_text"]

    print(text)
    return text

def generate_story(scenario):
    template = """
    You are a storyteller;
    You can generate the story based on simple narrative the story should not be more than 50 words;

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1),
        prompt=prompt,
        verbose=True
    )
    
    story = story_llm.predict(scenario = scenario)

    print(story)
    return(story)
#text to speech

def text2speech(message):
    
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer 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"}

    payloads = {
        "inputs" : message
    }
	
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file: 
        file.write(response.content)

scenario= img2txt("a.jpg")
story = generate_story(scenario)
text2speech(story)



