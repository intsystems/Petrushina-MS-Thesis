from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import numpy as np
import pandas as pd
import seaborn as sns

import re
from matplotlib import pyplot as plt

import os
from tqdm.notebook import tqdm

PROMPT1 = """[INST]
List 10 different facts (captions) about this picture. Give the facts only about the content of the image, where one fact describes one feature of the image.
To describe people, objects, and events, use information from Wikipedia for well-known entities.
In cases of ambiguity in the image, describe each part separately.
Include information about objects, subjects, environments, their descriptions, and actions. 
The format of the facts should be: subject predicate addition.
Follow this template:
1.
2.
3.
<image>[/INST]"""

PROMPT2 = """USER:\
<image>
List 10 different facts (captions) about this picture. Give the facts only about the content of the image and not about the style or origin, where one fact describes one feature of the image. \
To describe people, objects, and events, use information from Wikipedia for well-known entities. \
In cases of ambiguity in the image, describe each part separately. \
Include information about objects, subjects, environments, their descriptions, and actions. \
The format of the facts should be: subject predicate addition. \
Follow this template:
1.
2.
3.
ASSISTANT:"""

def get_atomic_facts(image, llava_processor, llava_model, prompt):
    inputs = llava_processor(prompt, image, return_tensors="pt").to(llava_model.device)
    output = llava_model.generate(**inputs, 
                                  max_new_tokens=256,
                                  pad_token_id=llava_processor.tokenizer.eos_token_id
                                 # do_sample=True,
                                 # num_beams=3
                                 )
    atomic_facts = llava_processor.batch_decode(output, skip_special_tokens=True)[0][len(prompt) - 5:].split("\n")[:10]

    atomic_facts = [re.sub('\d*\. ', '', f) for f in atomic_facts]
    return atomic_facts


def classify_pair(premise, hypothesis, aggregate_nli, tokenizer, model, device):
    input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
    output = model(input["input_ids"].to(device))
    prediction = torch.softmax(output["logits"][0], -1).tolist()
    label_names = ["entailment", "neutral", "contradiction"]
    prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
    
    return aggregate_nli(prediction)


def classify_facts(facts, aggregate_nli, tokenizer, model, device):
    min_score = 0
    min_facts = None
    out = []
    for f1 in facts:
        for f2 in facts:
            score = round(classify_pair(f1, f2, aggregate_nli, tokenizer, model, device))
            out.append(score)
            if score < min_score:
                min_score = score

    out_reshaped = np.array(out).reshape(len(facts), len(facts))
    out_df = pd.DataFrame(out_reshaped, index=facts, columns=facts)

    tril = np.tril(out_reshaped, -1)
    triu = np.triu(out_reshaped, 1).T
    minimum = np.where((tril < 0) & (triu < 0), tril + triu, 0)
    out_df = pd.DataFrame(minimum, index=facts, columns=facts)

    out_metric = minimum.mean()
    return out_df, out_metric, min_facts


def process_image(img, llava_processor, llava_model, prompt, aggregate_nli, tokenizer, model, device, classify, visualize):
    # img = Image.open(os.path.join(root, filename))
    atomic_facts = get_atomic_facts(img, llava_processor, llava_model, prompt)
    if classify:
        classified_facts, metric, min_facts = classify_facts(atomic_facts, aggregate_nli, tokenizer, model, device)
        
        if visualize:
            fig, axs = plt.subplots(ncols=2)
            fig.set_figwidth(15)
            imgplot = plt.imshow(img)
            sns.heatmap(classified_facts, annot=True, fmt=".2f", ax=axs[0])

        return metric
    else:
        return atomic_facts

    # return res


def run_image_evaluation(llava_processor, llava_model, nli_model, aggregate_nli, data_path=None, images=None, prompt=1, classify=True, visualize=False):
    if prompt == 1:
        prompt = PROMPT1
    else:
        prompt = PROMPT2

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(nli_model)
    model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(device)

    label_to_id = {
        'entailment': 1,
        'neutral': 0,
        'contradiction': -1
    }

    res = {}

    if images is not None:
        for i, img in tqdm(enumerate(images)):
            res[i] = process_image(img, llava_processor, llava_model, prompt, aggregate_nli, tokenizer, model, device, classify, visualize)

    elif data_path is not None:
        for root, dirnames, filenames in os.walk(data_path):
            for filename in tqdm(filenames):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img = Image.open(os.path.join(root, filename))
                    res[filename] = process_image(img, llava_processor, llava_model, prompt, aggregate_nli, tokenizer, model, device, classify, visualize)
    else:
        raise ValueError("No images")
    return res
