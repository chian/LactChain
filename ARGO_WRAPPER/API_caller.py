import requests

from getpass import getpass
import os

# OpenAI
from openai import OpenAI

# Ollama
from ollama import Client
from langchain_community.llms import Ollama

# ARGO
from ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from CustomLLM import ARGO_LLM, ARGO_EMBEDDING

def get_openai_llm():
    os.environ['OPENAI_API_KEY'] = getpass("Enter OpenAI API Key: ")
    return OpenAI()

def get_ollama_llm(model="llama3"):
    return Ollama(model=model)

def get_argo_llm(model_type='gpt4', temperature=0.5):
    argo_wrapper_instance = ArgoWrapper()
    print("Type of argo_wrapper_instance:", type(argo_wrapper_instance))
    return ARGO_LLM(argo=argo_wrapper_instance)

def get_vllm_api(endpoint_url,model="llama3"):
    class VLLMAPI:
        def __init__(self, endpoint_url, model):
            self.endpoint_url = endpoint_url
            self.model = model

        def invoke(self, prompt):
            response = requests.post(self.endpoint_url, json={"model": self.model, "prompt": prompt})
            #print("response.json:", response.json())
            return response.json()

    return VLLMAPI(endpoint_url, model)

class LLMWrapper:
    def __init__(self, llm, llm_type):
        self.llm = llm
        self.llm_type = llm_type

    def invoke(self, prompt):
        if self.llm_type == 'ollama':
            return self.llm.invoke(prompt)
        elif self.llm_type in ['openai', 'argo', 'vllm']:
            # Assuming these LLMs have a method to handle prompts directly
            return self.llm.invoke(prompt)
        else:
            raise NotImplementedError(f"The LLM type {self.llm_type} does not support invocation.")

def get_llm(source, model=None, temperature=None, endpoint_url=None):
    if source == 'openai':
        llm = get_openai_llm()
        llm_type = 'openai'
    elif source == 'ollama':
        llm = get_ollama_llm(model=model if model else "llama3")
        llm_type = 'ollama'
    elif source == 'argo':
        llm = get_argo_llm(model_type=model if model else 'gpt4', temperature=temperature if temperature is not None else 0.5)
        llm_type = 'argo'
    elif source == 'vllm':
        if not endpoint_url:
            raise ValueError("Endpoint URL must be provided for vLLM")
        llm = get_vllm_api(endpoint_url)
        llm_type = 'vllm'
    else:
        raise ValueError("Unsupported LLM source")
    
    return LLMWrapper(llm, llm_type)

# Example usage
llm_source = 'ollama'  # Can be 'openai', 'ollama', 'argo', or 'vllm'
endpoint_url = 'http://140.221.70.43:5005/llm/v1'  # Specify the API endpoint URL

from pydantic import ValidationError

#llm = get_llm(llm_source)
llm = get_llm(llm_source, model="llama3:70b",endpoint_url=endpoint_url)

prompt = "Translate the following English text to French: | Hello, how are you?"
response = llm.invoke(prompt)
print("Response from Llama Client:", response)