from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate
import numpy as np
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Union, Literal
import sys, os
from pydantic import Field
from pathlib import Path
sys.path.append('/lus/eagle/projects/FoundEpidem/bhsu/2024_research/LactChain')

from use_cases.mine.lactchain.config import BaseConfig
from use_cases.mine.lactchain.argo_wrapper import ArgoLLM

class LangchainConfig(BaseConfig): 
    model:Literal['gpt-3.5-turbo', 'gpt-4', 
                  'gpt-4-turbo', 'gpt-4-turbo-preview', 'gpt-4o',
                  'claude-3-opus-20240229', 'gemini-pro']=Field(
                      'gpt-4o', 
                      description='what kind of llm to use'
    )
    temperature:float=Field(
        0.0, 
        description='temperature response'
    )
    verbose:bool=Field(
        True, 
        description='whether or not llm should be verbose or not'
    )

class EmbeddingConfig(BaseConfig): 
    model:Literal['text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002', 
                  'voyage-2', 'voyage-large-2', 'voyage-code-2']=Field(
                      'text-embedding-ada-002', 
                      description='what kind of embedding model to use'
                      )
    dimensions:int=Field(
        1024, 
        description='Emebdding dimension'
    )
    chunk_size:int=Field(
        1000, 
        description='Maximum number of texts to embed in each batch'
    )

class ArgoConfig(BaseConfig): 
    model_type:Literal['gpt35', 'gpt4', 'gpt4turbo']=Field(
        'gpt4', 
        description='what kind of gpt model to use for argo'
    )
    temperature:float=Field(
        0.0, 
        description='temperature for argo llm'
    )
    top_p:float=Field(
        0.0001, 
        description='top_p for logit selection for argo llm'
    )

class GeneratorConfig(BaseConfig): 
    dotenv_path:Path=Field(
        Path.home() / '.env', 
        description='Path to the .env file. Contains API keys: '
        'OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY',
    )
    llmtype:Literal['langchain', 'argo']=Field(
        'argo', 
        description='which llm generator to choose: langchain or argo'
    )
    llmconfig:LangchainConfig=Field(
        default_factory=LangchainConfig, 
        description='configs for langchain LLM generators'
        )
    embedconfig:EmbeddingConfig=Field(
        default_factory=EmbeddingConfig, 
        description='configs for langchain embeddings'
    )
    argoconfig:ArgoConfig=Field(
        default_factory=ArgoConfig, 
        description='config for argo generators'
    )

class LangChainGenerator:
    """Create simple language chains for inference."""

    def __init__(self, config: GeneratorConfig) -> None:
        """Initialize the LangChainGenerator."""
        from langchain.chains.llm import LLMChain
        from langchain_community.chat_models import ChatOpenAI
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import GoogleGenerativeAI

        # Load environment variables from .env file containing
        # API keys for the language models
        load_dotenv(config.dotenv_path)
        # Define the possible chat models
        chat_models = {
            'gpt-3.5-turbo': ChatOpenAI,
            'gpt-4':ChatOpenAI, 
            'gpt-4-turbo':ChatOpenAI, 
            'gpt-4-turbo-preview':ChatOpenAI, 
            'gpt-4o':ChatOpenAI,
            'gemini-pro': GoogleGenerativeAI,
            'claude-3-opus-20240229': ChatAnthropic,
        }
        # Get the chat model based on the configuration
        chat_model = chat_models.get(config.llmconfig.model)
        if not chat_model:
            raise ValueError(f'Invalid chat model: {config.llmconfig.model}')

        # Initialize the language model
        if config.llmtype=='langchain':
            self.llm=chat_model(**config.llmconfig.model_dump())
        elif config.llmtype=='argo': 
            self.llm=ArgoLLM(**config.argoconfig.model_dump())
        else: 
            raise ValueError(f'Invalid Generator Type: {config.llmtype}')
    
        # Create the prompt template (input only)
        prompt = ChatPromptTemplate.from_template('{input}')

        # Initialize the chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=config.verbose,
        )

    def generate(self, prompts: str | list[str]) -> list[str]:
        """Generate response text from prompts.

        Parameters
        ----------
        prompts : str | list[str]
            The prompts to generate text from.

        Returns
        -------
        list[str]
            A list of responses generated from the prompts
            (one response per prompt).
        """
        # Ensure that the prompts are in a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Format the inputs for the chain
        inputs = [{'input': prompt} for prompt in prompts]

        # Generate the outputs
        raw_outputs = self.chain.batch(inputs)

        # Extract the text from the outputs
        outputs = [output['text'] for output in raw_outputs]
        return outputs
        
