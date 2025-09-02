import os

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from openai import AzureOpenAI

embeddings = AzureOpenAIEmbeddings(
    model=os.getenv("EMBEDDINGS_AZURE_OPENAI_MODEL_NAME"),
    azure_deployment=os.getenv("EMBEDDINGS_AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("EMBEDDINGS_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("EMBEDDINGS_AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("EMBEDDINGS_AZURE_OPENAI_API_VERSION"),
)

llm = AzureChatOpenAI(
    model=os.getenv("LLM_AZURE_OPENAI_MODEL_NAME"),
    azure_deployment=os.getenv("LLM_AZURE_OPENAI_DEPLOYMENT_NAME"),
    azure_endpoint=os.getenv("LLM_AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("LLM_AZURE_OPENAI_API_KEY"),
    openai_api_version=os.getenv("LLM_AZURE_OPENAI_API_VERSION"),
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

llm_openai = AzureOpenAI(
    api_key=os.getenv("LLM_AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("LLM_AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("LLM_AZURE_OPENAI_ENDPOINT"),
)
