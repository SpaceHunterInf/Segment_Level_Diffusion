from multiprocessing.spawn import prepare
import os
import json
import torch

from datasets import load_dataset, Value
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from functools import partial

def exists(x):
    return x is not None

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Parse each line as JSON and append to the list
            data.append(json.loads(line))
    return data

class sentence_dataset(Dataset):
    def __init__(self, data, max_input_sentences=2, max_output_sentences=2, sentence_pad_token='empty sentence',
                 parallel_encoding=False, parallel_decoding=False):
        self.data = data
        self.max_input_sentences = max_input_sentences
        self.max_output_sentences = max_output_sentences
        
        self.sentence_pad_token = sentence_pad_token
        self.parallel_encoding = parallel_encoding
        self.parallel_decoding = parallel_decoding
        
    
    def __getitem__(self, index):
        
        x = {}
        
        if self.parallel_encoding:
            x['src'] = self.data[index]['input'] # a list of sentences
        else:
            x['src'] = [self.data[index]['src']] # only one long sentence
        if self.parallel_decoding:
            x['trg'] = self.data[index]['output']
        else:
            x['trg'] = [self.data[index]['trg']]
        
        x['src_utt_mask'] = [1 for i in x['src']]
        x['trg_utt_mask'] = [1 for i in x['trg']]
        
        # mask = 0 if empty sentence else 1
        # to be multiplied with tokenizer to 0 out unnecessary sentences
        
        if self.parallel_encoding:
            if len(self.data[index]['input']) < self.max_input_sentences:
                pad_sentences = [self.sentence_pad_token for i in range(self.max_input_sentences - len(self.data[index]['input']))]
                x['src'] += pad_sentences
                zeros = [0 for i in pad_sentences]
                x['src_utt_mask'] += zeros
            else:
                x['src'] = x['src'][-self.max_input_sentences:]
                x['src_utt_mask'] = x['src_utt_mask'][-self.max_input_sentences:]

        if self.parallel_decoding:
            if len(self.data[index]['output']) < self.max_output_sentences:
                pad_sentences = [self.sentence_pad_token for i in range(self.max_output_sentences - len(self.data[index]['output']))]
                x['trg'] += pad_sentences
                zeros = [0 for i in pad_sentences]
                x['trg_utt_mask'] += zeros
            else:
                x['trg'] = x['trg'][:self.max_output_sentences]
                x['trg_utt_mask'] = x['trg_utt_mask'][:self.max_output_sentences]
        
        #make them 2D for row wise masking
        
        x['src_utt_mask'] = torch.tensor(x['src_utt_mask'], dtype=torch.int64).unsqueeze(-1)
        x['trg_utt_mask'] = torch.tensor(x['trg_utt_mask'], dtype=torch.int64).unsqueeze(-1)
        
        
        return x
    
    def __len__(self):
        return len(self.data)
        
def t5_sentence_collate_fn(data, tokenizer, max_input_seq_length, max_output_seq_length,
                           parallel_encoding=False, parallel_decoding=False, mask_mode='pad_token_fill'):
    
    def truncate_to_last_n_tokens(text, max_tokens=512):
        tokens = tokenizer.encode(text, add_special_tokens=False)
    
        # If length exceeds max_tokens, keep only the last max_tokens
        if len(tokens) > max_tokens:
            tokens = tokens[-max_tokens:]
        
        # Convert back to text
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=True)
        
        return truncated_text
    
    batch_data = {}
    for key in data[0]:
        batch_data[key] = [d[key] for d in data]
    
    batch_of_src_ids = []
    batch_of_src_attention_masks = []
    
    batch_of_trg_ids = []
    batch_of_trg_attention_masks = []
    
    batch_of_label_ids = []
    
    for i in range(len(batch_data['src'])):
        #For now, let's not disentangle the input, only the output
        #tokenized_src = tokenizer(batch_data['src'][i], padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        if not parallel_encoding:
            src = ' [UTT_BREAK] '.join(batch_data['src'][i])
        else:
            src = batch_data['src'][i]
        if not parallel_decoding:
            trg = ' [UTT_BREAK] '.join(batch_data['trg'][i])
        else:
            trg = batch_data['trg'][i]
        
        tokenized_src = tokenizer(truncate_to_last_n_tokens(src), padding='max_length', truncation=True, max_length=max_input_seq_length, return_tensors='pt')
        tokenized_trg = tokenizer(trg, padding='max_length', truncation=True, max_length=max_output_seq_length, return_tensors='pt')
        
        batch_of_src_ids.append(tokenized_src['input_ids'])
        batch_of_src_attention_masks.append(tokenized_src['attention_mask'])
        
        label_ids = tokenized_trg['input_ids'].clone()
        #breakpoint()
        if mask_mode == 'pad_token_fill':
            label_ids = label_ids.masked_fill(batch_data['trg_utt_mask'][i] == 0,  tokenizer.pad_token_id)
        else:
            raise NotImplementedError
        label_ids.masked_fill_(label_ids == tokenizer.pad_token_id, -100)
        
        batch_of_trg_ids.append(tokenized_trg['input_ids'])
        batch_of_trg_attention_masks.append(tokenized_trg['attention_mask'])
        batch_of_label_ids.append(label_ids)
        
    batch_data['cond_input_ids'] = torch.stack(batch_of_src_ids)
    batch_data['cond_attention_masks'] = torch.stack(batch_of_src_attention_masks)
    batch_data['input_ids'] = torch.stack(batch_of_trg_ids)
    batch_data['attention_masks'] = torch.stack(batch_of_trg_attention_masks)
    batch_data['label_ids'] = torch.stack(batch_of_label_ids)
    
    #breakpoint()
    
    return batch_data

def prepare_data(data_path, tokenizer, train_batch_size=4, val_batch_size=16, max_input_sentences=2, max_output_sentences=2, 
                 max_input_seq_length=512, max_output_seq_length=32,
                 sentence_pad_token='empty sentence', parallel_encoding=False, parallel_decoding=False):
    
    splits = ['train', 'valid', 'test', 'train_valid']
    data_loaders = []
    
    for s in splits:
        if s == 'train_valid':
            data = read_jsonl(os.path.join(data_path, 'train' + '.jsonl'))
            dataset = sentence_dataset(data[:1000], max_input_sentences=max_input_sentences, max_output_sentences=max_output_sentences, sentence_pad_token=sentence_pad_token,
                                       parallel_encoding=parallel_encoding, parallel_decoding=parallel_decoding)
        else:
            data = read_jsonl(os.path.join(data_path, s + '.jsonl'))
            dataset = sentence_dataset(data, max_input_sentences=max_input_sentences, max_output_sentences=max_output_sentences, sentence_pad_token=sentence_pad_token,
                                       parallel_encoding=parallel_encoding, parallel_decoding=parallel_decoding)
        data_loaders.append(DataLoader(dataset, 
                                       collate_fn=partial(t5_sentence_collate_fn, tokenizer=tokenizer, max_input_seq_length=max_input_seq_length, max_output_seq_length=max_output_seq_length,
                                                          parallel_decoding=parallel_decoding, parallel_encoding=parallel_encoding), 
                                       batch_size= (train_batch_size if 'train' in s else val_batch_size), 
                                       shuffle= (True if s=='train' else False), pin_memory=True, num_workers=0))
        
    train, valid, test, train_valid = data_loaders
    return train, valid, test, train_valid

def main():
    # a function testing whether utils work
    
    data_path = 'datasets/last_16_delibot_data'
    tokenizer = AutoTokenizer.from_pretrained('saved_latent_models/utt_delibot_outputs/2024-05-19_21-10-04/tokenizer/')
    
    train, valid, test, train_val = prepare_data(data_path, tokenizer, parallel_decoding=True, max_input_sentences=16, max_output_sentences=16)

    torch.manual_seed(42)
    for s in [train, valid, test, train_val]:
        for b in s:
            #print('input')
            #print(b['src'])
            print(tokenizer.batch_decode(b['cond_input_ids'].reshape(4, -1)))
            #print('output'
            #print(b['trg'])
            print(tokenizer.batch_decode(b['input_ids'].reshape(4, -1)))
            print(tokenizer.batch_decode(b['label_ids']))
            breakpoint()
    
if __name__ == "__main__":
    main()