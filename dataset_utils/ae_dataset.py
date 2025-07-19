from multiprocessing.spawn import prepare
import os
import json
import torch

from datasets import load_dataset, Value
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from functools import partial


def read_jsonl(file_path):
    """
    Reads a JSONL (JSON Lines) file and returns a list of dictionaries.

    Parameters:
    file_path (str): The path to the JSONL file.

    Returns:
    list: A list of dictionaries where each dictionary represents a JSON object from the file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

class utt_dataset(Dataset):
    def __init__(self, data, encoder_tokenizer, decoder_tokenizer, max_length=64, noiser=None, cse=False):
        self.data = data
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.max_length = max_length
        self.noiser = noiser
        self.cse = cse
        
    def __getitem__(self, index):
        # Get the text data at the specified index
        if 'src' in self.data[index].keys():
            text = self.data[index]['src']
        elif 'text' in self.data[index].keys():
            text = self.data[index]['text']
        
        # Tokenize the input text using the encoder tokenizer
        encoder_input = self.encoder_tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize the input text using the decoder tokenizer
        decoder_input = self.decoder_tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Apply noise to the encoder input IDs if a noiser is provided
        if self.noiser:
            noisy_encoder_input_ids = self.noiser(encoder_input['input_ids'].squeeze())
        else:
            noisy_encoder_input_ids = encoder_input['input_ids'].squeeze()
        
        # Prepare the dictionary for returning the data
        x = {
            'text':text,
            'encoder_input_ids': noisy_encoder_input_ids,
            'decoder_input_ids': decoder_input['input_ids'].squeeze(),
            'attention_mask': encoder_input['attention_mask'].squeeze(),
            'decoder_attention_mask': decoder_input['attention_mask'].squeeze()
        }
        
        if self.cse: #more stuff for contrastive learning
            x['has_similar'] = 'similar' in self.data[index].keys()
            x['has_contrastive'] = 'contrastive' in self.data[index].keys()
            
            item = self.data[index]
            # Add similar sentences if they exist
            if 'similar' in item.keys():
                similar_input = self.encoder_tokenizer.encode_plus(
                    item['similar'],
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                x['similar_ids'] = similar_input['input_ids']
                x['similar_attention_mask'] = similar_input['attention_mask']

            # Add contrastive sentences if they exist
            if 'contrastive' in item.keys():
                contrastive_input = self.encoder_tokenizer.encode_plus(
                    item['contrastive'],
                    padding='max_length',
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                x['contrastive_ids'] = contrastive_input['input_ids']
                x['contrastive_attention_mask'] = contrastive_input['attention_mask']
        return x

    def __len__(self):
        return len(self.data)

def collate_fn(batch, encoder_tokenizer=None, decoder_tokenizer=None, cse=False):
    # Extract sequences from the batch
    encoder_input_ids = [item['encoder_input_ids'] for item in batch]
    decoder_input_ids = [item['decoder_input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    decoder_attention_masks = [item['decoder_attention_mask'] for item in batch]
    
    # Pad sequences to the maximum length within the batch
    encoder_input_ids = torch.nn.utils.rnn.pad_sequence(encoder_input_ids, batch_first=True, padding_value=0)
    decoder_input_ids = torch.nn.utils.rnn.pad_sequence(decoder_input_ids, batch_first=True, padding_value=0)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    decoder_attention_masks = torch.nn.utils.rnn.pad_sequence(decoder_attention_masks, batch_first=True, padding_value=0)
    
    # Decode to text (ignoring the pad tokens)
    decoded_encoder_texts = [
        encoder_tokenizer.decode(ids, skip_special_tokens=True) 
        for ids in encoder_input_ids
    ]
    
    decoded_decoder_texts = [
        decoder_tokenizer.decode(ids, skip_special_tokens=True)
        for ids in decoder_input_ids
    ]
    
    labels = decoder_input_ids.clone()
    
    # Set padding tokens in labels to -100 to ignore them in the loss computation
    labels[labels == 0] = -100
    
    batch_dict = {
        'encoder_input_ids': encoder_input_ids,
        'decoder_input_ids': decoder_input_ids,
        'attention_mask': attention_masks,
        'decoder_attention_mask': decoder_attention_masks,
        'decoded_encoder_texts': decoded_encoder_texts,
        'decoded_decoder_texts': decoded_decoder_texts,
        'label_ids': labels
    }
    
    if cse:
        has_similar = torch.tensor([item['has_similar'] for item in batch])
        has_contrastive = torch.tensor([item['has_contrastive'] for item in batch])
                # Handle similar sentences if they exist
        if any(has_similar):
            similar_ids = [item['similar_ids'] for item in batch if 'similar_ids' in item]
            similar_attention_mask = [item['similar_attention_mask'] for item in batch if 'similar_attention_mask' in item]
            if similar_ids:
                batch_dict['similar_ids'] = torch.nn.utils.rnn.pad_sequence(similar_ids, batch_first=True, padding_value=0).squeeze(1)
                batch_dict['similar_attention_mask'] = torch.nn.utils.rnn.pad_sequence(similar_attention_mask, batch_first=True, padding_value=0).squeeze(1)

        # Handle contrastive sentences if they exist
        if any(has_contrastive):
            contrastive_ids = [item['contrastive_ids'] for item in batch if 'contrastive_ids' in item]
            contrastive_attention_mask = [item['contrastive_attention_mask'] for item in batch if 'contrastive_attention_mask' in item]
            if contrastive_ids:
                batch_dict['contrastive_ids'] = torch.nn.utils.rnn.pad_sequence(contrastive_ids, batch_first=True, padding_value=0).squeeze(1)
                batch_dict['contrastive_attention_mask'] = torch.nn.utils.rnn.pad_sequence(contrastive_attention_mask, batch_first=True, padding_value=0).squeeze(1)

    return batch_dict

def get_dataloader(data, encoder_tokenizer, decoder_tokenizer, batch_size=32, max_length=64, noiser=None, shuffle=True, cse=False):
    """
    Creates a DataLoader for the given dataset.

    Args:
    - data (list): List of datapoints, each being a dictionary with a 'text' field.
    - encoder_tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the encoder.
    - decoder_tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the decoder.
    - batch_size (int): Number of samples per batch to load.
    - max_length (int): Maximum sequence length for the tokenized inputs.
    - noiser (callable, optional): Function to add noise to the encoder input IDs.
    - shuffle (bool): Whether to shuffle the data at every epoch.

    Returns:
    - DataLoader: DataLoader for the dataset.
    """
    
    # Initialize the dataset
    dataset = utt_dataset(
        data=data,
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        max_length=max_length,
        noiser=noiser,
        cse=cse
    )

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, encoder_tokenizer=encoder_tokenizer, decoder_tokenizer=decoder_tokenizer, cse=cse)
    )

    return dataloader

def get_utt_dataloader(file_path, encoder_tokenizer, decoder_tokenizer, max_length=64, noiser=None, batch_size=32, dev_mode=False, cse=False):
    
    if '.jsonl' in file_path:
        data = read_jsonl(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    
    if dev_mode:
        data=data[:100]
    
    dataloader = get_dataloader(data, encoder_tokenizer, decoder_tokenizer, 
                                max_length=max_length, noiser=noiser, batch_size=batch_size, cse=cse)
    return dataloader

if __name__ == '__main__':
    from noiser import SubNoiser
    
    # Example usage - replace with your actual dataset path
    file_path = 'datasets/example_data/dev.json'  # Updated to use relative path
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    
    # Get special token IDs (example: pad_token_id, eos_token_id, bos_token_id)
    special_tokens = [
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.bos_token_id
    ]

    # Initialize the noiser with a 10% trigger chance, 15% substitution probability, a seed, and special tokens
    noiser = SubNoiser(vocab_size=tokenizer.vocab_size, trigger_chance=1, sub_prob=0.03, seed=2024, special_tokens=special_tokens)

    # Get DataLoader
    dataloader = get_dataloader(data, tokenizer, tokenizer, batch_size=2, max_length=64, noiser=noiser)

    # Iterate over DataLoader
    for batch in dataloader:
        print(batch)
        break  # Just show one batch for example