## [Blog post with more details](https://towardsdatascience.com/high-quality-sentence-paraphraser-using-transformers-in-nlp-c33f4482856f)

## High-quality diverse sentence Paraphraser.

This paraphraser is trained on custom dataset with pairs of paraphrased sentences that are diverse. Diverse here means that pairs of sentences are selected such that there is significant difference in word order or at least the paraphrased output differs by multiple word changes. T5 large model from Huggingface is used to train the paraphraser.

<img src= './Diverse Sentence Paraphraser.png' > 



##  Google Colab Paraphraser Complete Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16zHH-g9z5S_gUQQk7vOxUfpT085wTQ9Z?usp=sharing)

## 1. Installation
```
!pip install transformers==4.10.2
!pip install sentencepiece==0.1.96
```

## 2. Running the code
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)


# Diverse Beam search

context = "Once, a group of frogs was roaming around the forest in search of water."
text = "paraphrase: "+context + " </s>"

encoding = tokenizer.encode_plus(text,max_length =128, padding=True, return_tensors="pt")
input_ids,attention_mask  = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

model.eval()
diverse_beam_outputs = model.generate(
    input_ids=input_ids,attention_mask=attention_mask,
    max_length=128,
    early_stopping=True,
    num_beams=5,
    num_beam_groups = 5,
    num_return_sequences=5,
    diversity_penalty = 0.70

)

print ("\n\n")
print ("Original: ",context)
for beam_output in diverse_beam_outputs:
    sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
    print (sent)
```
<details>
<summary>Show Output</summary>

```
Original:  Once, a group of frogs was roaming around the forest in search of water.
paraphrasedoutput: A herd of frogs was wandering around the woods in search of water.
paraphrasedoutput: A herd of frogs was wandering around the woods in search of water.
paraphrasedoutput: A gang of frogs was wandering around the forest in search of water at one time.
paraphrasedoutput: A herd of frogs was swaning around the woods in search of water.
paraphrasedoutput: A gang of frogs was roaming about the woods in search of water once more.

```
</details>

Check out more examples in the Colab Notebook.

## Try advanced question generation models for free:  https://questgen.ai/  


This paraphraser is released by QuestgenAI as a part of its open-source initiative to build advanced Question generation and related NLP algorithms. It is on a quest build the world's most advanced question generation AI leveraging on state-of-the-art transformer models like GPT-3, T5, BERT and OpenAI GPT-2 etc. <br>
We would appreciate if you can spread the word out about Questgen.
