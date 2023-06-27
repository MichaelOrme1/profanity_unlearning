from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration
from transformers import AdamW
from datasets import Dataset, DatasetDict
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base", model_max_length=512)

# Load the pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base").to(device)

def remove_punc(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char  # space is also a character
    return no_punct.lower()

def load_dataset(file_path):
    with open(file_path, "r") as file:
    
        data = {"input": [], "output": []}
        for line in file:
            line = line.strip()
            if line:
                qa_pairs = line.split('\t')
                input_seq = remove_punc(qa_pairs[0])
                output_seq = remove_punc(qa_pairs[1]) 
                data["input"].append(input_seq)
                data["output"].append(output_seq)
                
    return Dataset.from_dict(data)




def tokenization(example):
    encoded = tokenizer(example["input"], example["output"], padding="max_length", truncation=True)
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded





# Load the dataset
dataset = load_dataset("formatted_movie_lines.txt")

dataset = dataset.map(tokenization, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


training_args = Seq2SeqTrainingArguments(
    output_dir="output_dir",  # Directory to save checkpoints and logs
    evaluation_strategy="epoch",  # Evaluate model every epoch
    learning_rate=1e-4,  # Learning rate for training
    per_device_train_batch_size=4,  # Batch size per device during training
    per_device_eval_batch_size=4,  # Batch size per device during evaluation/validation
    num_train_epochs=5,  # Total number of training epochs
    logging_dir="logs",  # Directory to save logs
    save_total_limit=1,  # Maximum number of checkpoints to save
    save_strategy="epoch"  # Save checkpoints every epoch
)
trainer = Seq2SeqTrainer(
    model=model,  # The Seq2Seq model to train
    args=training_args,  # Training arguments
    train_dataset=dataset,  # Training dataset
    tokenizer=tokenizer  # Tokenizer for encoding the data
    
    
)

trainer.train()

