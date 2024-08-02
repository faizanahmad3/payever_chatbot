import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel


class CustomGPT2Instruct(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomGPT2Instruct, self).__init__()
        self.gpt2 = pretrained_model

        # Freeze GPT-2 instruct layers
        for param in self.gpt2.parameters():
            param.requires_grad = False

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
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt, max_length=300):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    output = model.gpt2.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


def response_gpt2(prompt):
    model_path = "../training/finetuned_custom_gpt2_instruct_model.pth"
    model, tokenizer, device = initialize_model(model_path)

    # Verify if the model is loaded correctly
    assert all(param.requires_grad == False for param in
               model.gpt2.parameters()), "GPT-2 model parameters are not frozen as expected"
    assert any(param.requires_grad for param in
               model.new_embeddings.parameters()), "New embeddings layer parameters are not trainable"
    assert any(
        param.requires_grad for param in model.adapter.parameters()), "Adapter layer parameters are not trainable"

    # Example usage
    generated_text = generate_text(model, tokenizer, device, prompt)
    print(f"Generated text:\n{generated_text}")
    return generated_text


# prompt = "how to create account on payever in detail"
# response_gpt2(prompt)
