# -*- coding: utf-8 -*-
#  
### PaliGemma Fine-tuning
#

# # # Initialization

#import libraries

import os
from PIL import Image
from pathlib import Path
import string
import random
from transformers import Trainer
import numpy as np
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from transformers import PaliGemmaProcessor
import torch

os.environ["HF_TOKEN"] = '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' #hugging face token
device = "cuda"

#  
# # # Load data. 
folder_names = ['Black', 'Green', 'Blue', 'TTR']
garbage_types = ['black', 'green', 'blue', 'other'] #Corresponds to ['landfill', 'compostable', 'recyclable', 'other']

#Function to load and format the data as needed
def format_data(i,split): 
    category = folder_names[i]
    answer = garbage_types[i]
    directory = "/home/ziheng.chang/garbage_data/garbage_data/CVPR_2024_dataset_"+split+'/'+category
    formatted_data = []
    for filename in sorted(os.listdir(directory)):
        image = (Image.open(directory+'/'+filename)).resize((224,224))
        description = ((Path(filename).stem).rstrip(string.digits)).replace('_',' ')
        label = answer
        formatted_entry = {'multiple_choice_answer': label,'question': f'What type of garbage is this {description}?','image': image
        }
        formatted_data.append(formatted_entry)

    # Example of accessing a formatted entry
    print(formatted_data[0])  # Should print the dictionary format for the first entry
    return formatted_data

#Load the train and evaluation data
train_data = format_data(0,'Train')+format_data(1,'Train')+format_data(2,'Train')+format_data(3,'Train')
random.shuffle(train_data)
val_data = format_data(0,'Val')+ format_data(1,'Val')+ format_data(2,'Val')+ format_data(3,'Val')
random.shuffle(val_data)
print(len(train_data),len(val_data))


# # # Preprocess data 
# Load the processor to preprocess the dataset.
model_id = "google/paligemma-3b-pt-224"
processor = PaliGemmaProcessor.from_pretrained(model_id)

#  
# Function to reprocess the dataset with a prompt template, pass it with batches of images to processor to convert to tokens.
def collate_fn(examples):
  texts = ["<image> <bos> answer " + example["question"] for example in examples]
  labels= [example['multiple_choice_answer'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens


# # # Load the model
# LoRA & QLoRA fine-tuning to reduce cost of training
bnb_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id,
                                                          quantization_config=bnb_config,
                                                          device_map={"":0})
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# # # Setup training, then train
#Initialize the `TrainingArguments`.

# Reasonable parameters are chosen after experimentation.
# 12 epochs gives good results without running for too long.
# 8 batch size gives good speed withou OOM
# learning rate of 4e-5 gives best accuracy
from transformers import TrainingArguments
args=TrainingArguments(
            num_train_epochs=12, 
            remove_unused_columns=False,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8, 
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=0.00004,
            weight_decay=1e-6,
            adam_beta2=0.999,
            optim="adamw_hf",
            save_strategy="epoch",
            eval_strategy="epoch",
            push_to_hub=True,
            save_total_limit=1,
            output_dir="paligemma_vqav2_2",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            eval_do_concat_batches = False
        )

 
# We can now start training.
trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=collate_fn,
        args=args
        )

trainer.train()

#Push the model to huggingface hub for future use.
#Also save it locally just in case
trainer.push_to_hub('palicoqiqi/paligemma_VQAv2_enel645_2')
trainer.save_model("model/paligemma_model_2")

