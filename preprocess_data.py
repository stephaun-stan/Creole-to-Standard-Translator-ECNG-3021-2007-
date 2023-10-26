import pandas as pd
from transformers import BartTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, AdamW
from tqdm import tqdm

# Load your CSV file
csv_file_path = 'creole_dataset.csv'
df = pd.read_csv(csv_file_path)

# Initialize BART tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

"""This function takes Creole and English text, tokenizes them, and pads the sequences to the maximum length. 
It returns a dictionary containing the original texts along with the input and label tokens."""
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

# Save the preprocessed data to a new CSV file containg input and label tokens
preprocessed_csv_path = 'creole_df.csv'
pd.DataFrame(preprocessed_data.tolist()).to_csv(preprocessed_csv_path, index=False)
 
print(f"Preprocessed data saved to {preprocessed_csv_path}")


