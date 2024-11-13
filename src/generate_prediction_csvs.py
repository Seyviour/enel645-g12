import csv
from csv import DictWriter
import os
from huggingface_hub import login
import torch
from PIL import Image
from transformers import AutoTokenizer, PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers import BitsAndBytesConfig
from pathlib import Path
import string
import random
from sklearn import metrics
# import matplotlib.pyplot as plt
import re
from torch.utils.data import Dataset, ConcatDataset
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM


RANDOM_SEED = 42
UAT = "" #Set hugging face access token here

# HUGGINGFACE_USER_NAME = "palicoqiqi"
HUGGINGFACE_USER_NAME = "seyviour"

IS_TEST_JOB = False
# IS_TEST_JOB = True

UAT = UAT or os.getenv("HF_TOKEN")
if UAT is None:
    raise Exception("Hugging Face Token is not set")
os.environ["HF_TOKEN"] = UAT #hugging face token
login(token=UAT)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ImageTextDataset(Dataset):
    def __init__(self, base_folder, sub_folder, label):

        self.base_folder = base_folder
        self.sub_folder = sub_folder
        self._path = os.path.join(self.base_folder, self.sub_folder)
        self._file_names = sorted(os.listdir(self._path))
        self.label = label

    def _get_image_path(self, idx):
        full_path = os.path.join(self._path, self._file_names[idx])
        return full_path

    def __len__(self):
        return len(self._file_names)

    def to_description(self, file_name):
        file_name_no_ext, _ = os.path.splitext(file_name)
        text = file_name_no_ext.replace('_', ' ')
        text_without_digits = re.sub(r'\d+', '', text)
        return text_without_digits.strip()

    def _get_image_at_idx(self, idx):
        filepath = self._get_image_path(idx)
        image = Image.open(filepath).convert('RGB')  # Convert to RGB
        return image

    def _get_text_at_idx(self, idx):
        true_idx = idx % (len(self._file_names))
        filename = self._file_names[true_idx]
        text = self.to_description(filename)
        return text

    def __getitem__(self, idx):
        # Returns a dictionary containing the image, question, and label for the specified index.
        description = self._get_text_at_idx(idx)
        image = self._get_image_at_idx(idx)
        label = self.label
        path = os.path.join(self.sub_folder, self._file_names[idx])
        data_point = {
            'multiple_choice_answer': label,
            'question': f'What type of garbage is this {description}?',
            'image': image,
            'path': path
        }
        return data_point

def make_garbage_dataset(basefolder:str)->tuple[ConcatDataset, dict]:
    # Ensure folders are sorted, exclude hidden files/folders
    class_folders = sorted([x for x in os.listdir(basefolder) if x[0]!='.'])
    individual_datasets = [ImageTextDataset(basefolder, x, x.lower()) for x in class_folders]
    return ConcatDataset(individual_datasets)

model_id = "palicoqiqi/paligemma_vqav2_2"
processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


folder_names = ['Black', 'Green', 'Blue', 'TTR']
garbage_types = ['black', 'green', 'blue', 'other']


OUTPUT_PATH = "prediction_csvs"
TEST_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"
TEST_PATH = "/Users/tomi_owolabi/garbage_data/garbage_data/CVPR_2024_dataset_Test"

test_set = make_garbage_dataset(TEST_PATH)

def make_predictions(examples, file_path):
    field_names = ["path", "predicted", "label"]
    feedback_file_handle = open(file_path, "+a", newline='')
    csv_writer = csv.DictWriter(feedback_file_handle, field_names)
    csv_writer.writeheader()
    count = 0 
    for example in examples:
        count+=1
        texts = "<image> <bos> answer " + example["question"]
        label = example['multiple_choice_answer']
        image = example["image"]
        
        inputs = processor(text=texts, images=image, padding="longest", do_convert_rgb=True, return_tensors="pt").to(device)
        inputs = inputs.to(dtype=model.dtype)
        
        with torch.no_grad():
            output = model.generate(**inputs, max_length=496)

        output = processor.decode(output[0][-2], skip_special_tokens=True)
        output = output.lower()
        output = output.strip()

        row = { 
            "path": example["path"],
            "label": label,
            "predicted": output
        }
        csv_writer.writerow(row)
        if (count%100):
            print(count)
    # print('Completed')
    # return actual, predicted

make_predictions(test_set, "predictions.csv")




