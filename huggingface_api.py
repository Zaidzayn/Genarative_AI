from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

# STEP 1: Create the endpoint LLM with chat-compatible model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=100,
    temperature=0.7,
)

# STEP 2: Wrap it inside ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# STEP 3: Use invoke
response = chat_model.invoke("What is the capital of France?")
print(response)
