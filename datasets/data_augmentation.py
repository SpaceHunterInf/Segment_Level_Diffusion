from vllm import SamplingParams, LLM
import json
from pathlib import Path
from tqdm import tqdm
import logging

def extract_between_markers(text):
    try:
        start = text.index("<<<") + 3
        end = text.index(">>>", start)
        return text[start:end].strip()
    except ValueError:
        return None

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def generate_paraphrases(model, data):
    prompt_template = """Paraphrase the following sentence. Return only the paraphrased text bounded by <<< and >>> for easy parsing.

Example Sentence: Jamie gets married and they spent the rest of their lives together.
Example Output: <<<Jamie gets married, and they remain together for the rest of their lives.>>>

Input Sentence: {sentence}

Output Sentence: """

    prompts = [prompt_template.format(sentence=item['text']) for item in data]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)
    outputs = model.generate(prompts, sampling_params)
    
    for item, output in zip(data, outputs):
        paraphrase = extract_between_markers(output.outputs[0].text)
        item['similar'] = paraphrase if paraphrase else item['text']
    
    return data

def main():
    
    output_dir = Path('ecqa_utt_with_final')
    output_dir.mkdir(exist_ok=True)
    
    model = LLM(model='meta-llama/Meta-Llama-3-8B-Instruct', dtype='bfloat16')
    
    for split in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
        input_path = Path('your_input_path') / split
        output_path = Path('your_output_path') / split
        
        data = read_jsonl(input_path)
        processed_data = generate_paraphrases(model, data)
        write_jsonl(processed_data, output_path)
        print(f"Processed {split}: {len(processed_data)} examples")

if __name__ == '__main__':
    main()