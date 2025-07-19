import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional
import random

class Seq2SeqContrastiveDataset(Dataset):
    def __init__(
        self, 
        data: List[Dict],
        tokenizer,
        max_length: int = 512
    ):
        """
        Args:
            data: List of dictionaries containing 'input' and 'output' keys,
                 optionally 'similar' or 'contrastive' keys
            tokenizer: Tokenizer object with encode method
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input and output
        input_ids = self.tokenizer.encode(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).squeeze(0)
        
        output_ids = self.tokenizer.encode(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).squeeze(0)

        sample = {
            'input_ids': input_ids,
            'output_ids': output_ids,
            'has_similar': 'similar' in item,
            'has_contrastive': 'contrastive' in item
        }

        # Add similar sentences if they exist
        if 'similar' in item:
            similar_ids = self.tokenizer.encode(
                item['similar'],
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).squeeze(0)
            sample['similar_ids'] = similar_ids

        # Add contrastive sentences if they exist
        if 'contrastive' in item:
            contrastive_ids = self.tokenizer.encode(
                item['contrastive'],
                max_length=self.max_length,
                truncation=True,
                return_tensors='pt'
            ).squeeze(0)
            sample['contrastive_ids'] = contrastive_ids

        return sample

def collate_fn(batch: List[Dict]):
    """
    Custom collate function to handle variable length sequences and optional fields
    """
    # Collect all available fields
    input_ids = [item['input_ids'] for item in batch]
    output_ids = [item['output_ids'] for item in batch]
    
    # Get masks for samples with similar/contrastive sentences
    has_similar = torch.tensor([item['has_similar'] for item in batch])
    has_contrastive = torch.tensor([item['has_contrastive'] for item in batch])

    # Pad sequences
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    output_ids = pad_sequence(output_ids, batch_first=True, padding_value=0)
    
    batch_dict = {
        'input_ids': input_ids,
        'output_ids': output_ids,
        'has_similar': has_similar,
        'has_contrastive': has_contrastive
    }

    # Handle similar sentences if they exist
    if any(has_similar):
        similar_ids = [item['similar_ids'] for item in batch if 'similar_ids' in item]
        if similar_ids:
            similar_ids = pad_sequence(similar_ids, batch_first=True, padding_value=0)
            batch_dict['similar_ids'] = similar_ids

    # Handle contrastive sentences if they exist
    if any(has_contrastive):
        contrastive_ids = [item['contrastive_ids'] for item in batch if 'contrastive_ids' in item]
        if contrastive_ids:
            contrastive_ids = pad_sequence(contrastive_ids, batch_first=True, padding_value=0)
            batch_dict['contrastive_ids'] = contrastive_ids

    return batch_dict

# Example usage
def create_dataloader(
    data: List[Dict],
    tokenizer,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
):
    """
    Create a DataLoader with the custom dataset and collate function
    """
    dataset = Seq2SeqContrastiveDataset(data, tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    