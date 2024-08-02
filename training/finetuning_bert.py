import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Custom model with frozen GPT-2 instruct layers and new layers
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


# Load and preprocess data
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.inputs = []
        self.labels = []
        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.inputs.append(torch.tensor(encodings["input_ids"]))
            self.labels.append(torch.tensor(encodings["input_ids"]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


print("Loading data...")
# Load data from CSV
df = pd.read_csv('../normalizing_data/normalized_support_data.csv')
texts = df["text_normalized"].tolist()

print("Splitting data...")
# Split data
train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)

print("Initializing tokenizer and model...")
# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
model = CustomGPT2Instruct(gpt2_model)

print("Preparing datasets...")
# Prepare datasets
max_length = 128
train_dataset = TextDataset(train_texts, tokenizer, max_length)
val_dataset = TextDataset(val_texts, tokenizer, max_length)

print("Creating data loaders...")
# Create data loaders
batch_size = 4
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()
num_epochs = 5

print("Starting training...")
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch in progress_bar:
        inputs, labels = [b.to(device) for b in batch]
        outputs = model(inputs)
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"loss": loss.item()})

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = [b.to(device) for b in batch]
            outputs = model(inputs)
            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

print("Saving model...")
# Save the fine-tuned model
torch.save(model.state_dict(), "finetuned_custom_gpt2_instruct_model.pth")

print("Generating text...")


# Generate text
def generate_text(model, prompt, max_length=100):
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
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Example usage
prompt = "create an account on payever"
generated_text = generate_text(model, prompt)
print(f"Generated text:\n{generated_text}")

print("Generating text with original model...")


# Generate text using only the original GPT-2 instruct model
def generate_text_original_gpt2_instruct(prompt, max_length=100):
    model.gpt2.eval()
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
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# Example usage of original GPT-2 instruct generation
original_generated_text = generate_text_original_gpt2_instruct(prompt)
print(f"Original GPT-2 instruct generated text:\n{original_generated_text}")

print("Done!")
