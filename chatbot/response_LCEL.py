import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.runnables import RunnablePassthrough
import logging


class CustomGPT2Instruct(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomGPT2Instruct, self).__init__()
        self.gpt2 = pretrained_model

        # Freeze GPT-2 instruct layers
        for param in self.gpt2.parameters():
            param.requires_grad = True

        # Add new trainable layers
        self.new_embeddings = nn.Embedding(self.gpt2.config.vocab_size, self.gpt2.config.n_embd)
        self.adapter = nn.Sequential(
            nn.Linear(self.gpt2.config.n_embd, self.gpt2.config.n_embd // 2),
            nn.ReLU(),
            nn.Linear(self.gpt2.config.n_embd // 2, self.gpt2.config.n_embd)
        )

    def forward(self, input_ids, attention_mask=None):
        # Use new embeddings
        embeddings = self.new_embeddings(input_ids)
        adapted = self.adapter(embeddings)

        # Pass through frozen GPT-2 instruct
        outputs = self.gpt2(inputs_embeds=adapted, attention_mask=attention_mask)
        return outputs


def initialize_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = CustomGPT2Instruct(gpt2_model)

    # Load the fine-tuned model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    return model, tokenizer, device


# def save_model_and_tokenizer(model, tokenizer, directory):
#     # Save only the GPT-2 part of the model
#     gpt2_model = model.gpt2
#     gpt2_model.save_pretrained(directory)
#     tokenizer.save_pretrained(directory)

def lcel_chain(input_text, retriever):
    try:
        # Initialize your model and tokenizer
        logging.info("initializing model")
        model_path = "../training/finetuned_custom_gpt2_instruct_model.pth"
        model, tokenizer, device = initialize_model(model_path)
        logging.info("model loaded")

        # Save the fine-tuned model and tokenizer in a temporary directory for Hugging Face compatibility
        # temp_dir = "./temp_model"
        # os.makedirs(temp_dir, exist_ok=True)
        # save_model_and_tokenizer(model, tokenizer, temp_dir)

        # Use Hugging Face's pipeline with LangChain
        hf_pipeline = pipeline("text-generation", model=model.gpt2, tokenizer=tokenizer,
                               device=0 if torch.cuda.is_available() else -1)
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

        prompt_template = PromptTemplate(template="response in polite manner of given question: {question}")
        rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt_template
                | llm
                | StrOutputParser()
        )
        result = rag_chain.invoke(input_text)
        return result

    except Exception as e:
        logging.error(e)
