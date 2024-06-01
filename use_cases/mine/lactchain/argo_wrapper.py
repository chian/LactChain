import os
import requests
import json
from typing import Any, List, Mapping, Optional, Tuple, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import requests
import json
import os
from pydantic import Field, BaseModel

from enum import Enum

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate

MODEL_GPT35 = "gpt35"
MODEL_GPT4 = "gpt4"

class ArgoWrapper:
    default_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

    def __init__(self, 
                 url = None, 
                 model = MODEL_GPT35, 
                 system = "",
                 temperature = 0.8, 
                 top_p=0.7, 
                 user = os.getenv("USER"))-> None:
        self.url = url
        if self.url is None:
            self.url = ArgoWrapper.default_url
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.user = user
        self.system = ""

    def invoke(self, prompt: str):
        headers = {
            "Content-Type": "application/json"
        }
        data = {
                "user": self.user,
                "model": self.model,
                "system": self.system,
                "prompt": [prompt],
                "stop": [],
                "temperature": self.temperature,
                "top_p": self.top_p
        }
            
        data_json = json.dumps(data)    
        response = requests.post(self.url, headers=headers, data=data_json)

        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

class ArgoEmbeddingWrapper:
    default_url = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"

    def __init__(self, url = None, user = os.getenv("USER")) -> None:
        self.url = url if url else ArgoEmbeddingWrapper.default_url
        self.user = user
        #self.argo_embedding_wrapper = argo_embedding_wrapper

    def invoke(self, prompts: list):
        headers = { "Content-Type": "application/json" }
        data = {
            "user": self.user,
            "prompt": prompts
        }
        data_json = json.dumps(data)
        response = requests.post(self.url, headers=headers, data=data_json)
        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code}")

    def embed_documents(self, texts):
        return self.invoke(texts)

    def embed_query(self, query):
        return self.invoke(query)[0]


################ applying the argo wrapper to make a simple langchain model ########################
class ModelType(Enum):
    GPT35 = 'gpt35'
    GPT4 = 'gpt4'

class ArgoLLM(LLM):

    model_type: ModelType = ModelType.GPT4
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.0
    system: Optional[str]
    top_p: Optional[float]= 0.0000001
    user: str = os.getenv("USER")
    
    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        headers = {
            "Content-Type": "application/json"
        }
        params = {
            **self._get_model_default_parameters,
            **kwargs,
            "prompt": [prompt],
            "stop": []
        }

        params_json = json.dumps(params);
        # print(params_json)
        response = requests.post(self.url, headers=headers, data=params_json)

        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed['response']
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _get_model_default_parameters(self):
        return {
            "user": self.user,
            "model": self.model,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p":  self.top_p
        }

    @property
    def model(self):
        return self.model_type.value
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

# orig ArgoEmbeddingWrapper
class ArgoEmbedder(Embeddings):
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"
    user: str = os.getenv("USER")

    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self, 
        prompts: List[str], 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        headers = { "Content-Type": "application/json" }
        params = { 
            "user": self.user, 
            "prompt": prompts
        }
        params_json = json.dumps(params)
        response = requests.post(self.url, headers=headers, data=params_json)
        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

    def embed_documents(self, texts):
        return self.invoke(texts)

    def embed_query(self, query):
        return self.invoke(query)[0]