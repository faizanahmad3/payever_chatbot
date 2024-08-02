from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from embedding_model import embedding_model
from response_LCEL import lcel_chain
import logging

emb_dir = "../chatbot/chroma_embeddings"


def csv_loader(path):
    logging.info("csv loading")
    loader = CSVLoader(path, encoding='utf-8')
    data = loader.load()
    logging.info("csv loaded")
    return data


# def split_text(csv_text, chunksize=700, chunkoverlap=70):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunksize,
#         chunk_overlap=chunkoverlap,
#         length_function=len,
#         separators=[]
#     )
#     text_chunks = text_splitter.split_text(csv_text)
#     return text_chunks


def documents_ingestor(doc_splits):
    try:
        logging.info("documents ingestion started")
        Chroma.from_documents(
            documents=doc_splits,
            collection_name="payever_documentation_emb",
            embedding=embedding_model,
            persist_directory=emb_dir
        )
        logging.info("documents ingestion completed")
    except Exception as e:
        logging.error(e)


def document_retriever():
    try:
        vectorstore = Chroma(
            persist_directory=emb_dir,
            embedding_function=embedding_model
        )
        return vectorstore.as_retriever(search_type="similarity")
    except Exception as e:
        logging.error(e)


def payever_gpt2(input_text: str, file_loader: bool, ingestion: bool):
    if file_loader:
        path = 'normalized_data.csv'
        docs = csv_loader(path)
        logging.info(docs[0])
        if ingestion:
            documents_ingestor(docs)  # ingest only one time

    retriever = document_retriever()
    answer = lcel_chain(input_text, retriever)
    print(answer)
    return answer
