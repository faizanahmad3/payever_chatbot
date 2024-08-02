from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embedding_model_name = "BAAI/bge-large-en-v1.5"
embedding_model_kwargs = {'device': 'cpu'}
embedding_encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding_model = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)
