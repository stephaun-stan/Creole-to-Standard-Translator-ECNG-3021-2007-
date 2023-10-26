import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, AdamW
from tqdm import tqdm
import torch
import pandas as pd
from transformers import BartTokenizer

# Load your CSV file
csv_file_path = 'creole_df.csv'
df = pd.read_csv(csv_file_path)

# Initialize BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def preprocess_data(creole_text, english_text, max_length=128):
    creole_tokens = tokenizer.encode(creole_text, return_tensors='pt', truncation=True)
    english_tokens = tokenizer.encode(english_text, return_tensors='pt', truncation=True)

    # Determine the maximum sequence length in the batch
    pad_length = max(creole_tokens.shape[1], english_tokens.shape[1], max_length)

    # Pad sequences to the same length
    creole_tokens = torch.nn.functional.pad(creole_tokens, (0, pad_length - creole_tokens.shape[1]))
    english_tokens = torch.nn.functional.pad(english_tokens, (0, pad_length - english_tokens.shape[1]))

    return {
        'creole_text': creole_text,
        'english_text': english_text,
        'input_ids': creole_tokens.flatten(),
        'labels': english_tokens.flatten()
    }


# Apply preprocessing to each row in the DataFrame
preprocessed_data = df.apply(lambda row: preprocess_data(row['Creole'], row['English']), axis=1)

# Save the preprocessed data to a new CSV file
preprocessed_csv_path = 'creole_df.csv'
pd.DataFrame(preprocessed_data.tolist()).to_csv(preprocessed_csv_path, index=False)

print(f"Preprocessed data saved to {preprocessed_csv_path}")

# Define a custom dataset class
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'labels': self.data[idx]['labels']
        }

# Function to train the BART model
def train_bart_model(preprocessed_data, num_epochs=6, learning_rate=5e-5, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load BART model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)

    # Prepare DataLoader
    dataset = TranslationDataset(preprocessed_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Set up optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')

    # Save the trained model
    model.save_pretrained('trained_bart_model')
    tokenizer.save_pretrained('trained_bart_tokenizer')


    print('Training complete. Model saved to "trained_bart_model" directory.')

# Call the function to train the BART model
train_bart_model(preprocessed_data)