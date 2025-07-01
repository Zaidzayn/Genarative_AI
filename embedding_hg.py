from langchain_huggingface import HuggingFaceEmbeddings


embeddings= HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Delhi is the capital of India."

embeddings_result = embeddings.embed_query(text)
print(str(embeddings_result))    
