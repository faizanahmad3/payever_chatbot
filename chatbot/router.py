from fastapi import APIRouter
from utilities import csv_loader, documents_ingestor, document_retriever, payever_gpt2
router = APIRouter()
router.add_api_route("/OK", lambda: "OK", methods=["GET"])
router.add_api_route("/GPT2_Response/", payever_gpt2, methods=["POST"])
