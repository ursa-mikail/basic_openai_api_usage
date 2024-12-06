# Install the Transformers, Datasets, and Evaluate libraries
!pip install datasets evaluate transformers[sentencepiece]
!pip install faiss-gpu
!pip install -U sentence-transformers
!pip install openai

"""
Resume analysis with GPT-3

GPT-3 demonstrated to give answers to questions on information publicly available on the web, 
in books, and in all the vast sources of information used to train these models. However, 
some use cases require us to fine-tune the models to get the best results. 

In this example usage, it explores how to use GPT-3's Question And Answers capabilities in a real-world 
scenario, analyzing resumes.
"""

from transformers.pipelines import pipeline
import os, requests
from datasets import Dataset

question_answerer = pipeline('question-answering')

# 1. Download and Load Data
url = 'https://raw.githubusercontent.com/raghavendranhp/Resume_screening/refs/heads/main/UpdatedResumeDataSet.csv'
save_dir = './sample_data/'
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, 'UpdatedResumeDataSet.csv')

response = requests.get(url)
response.raise_for_status()

with open(save_path, 'wb') as file:
    file.write(response.content)


import pandas as pd
df = pd.read_csv("./sample_data/UpdatedResumeDataSet.csv")
resume_dataset = Dataset.from_pandas(df)
resume_dataset

resume_dataset.set_format("pandas")
df = resume_dataset[:]
df.head()

#!pip install openai

import pandas as pd
import openai
import numpy as np
import pickle

COMPLETIONS_MODEL = "text-babbage-001"

openai.api_key = input('Enter your OpenAi key: ')

context = """
Candidate
Juan is a Software engineer ...
Career History:

November 2019 — Current
Senior Software Engineer maint... 
"""

# Function to create context from DataFrame
def create_context_from_df(df):
    context = ""
    for _, row in df.iterrows():
        candidate = f"Candidate\n{row['Name']} is a {row['Designation']} with experience in {row['Experience']}. Proficient in: {row['Skills']}\n\n"
        career_history = row['Description'].replace('. ', '.\n')
        context += candidate + "Career History:\n" + career_history + "\n---\n"
    return context

# Generate context from the DataFrame
context = create_context_from_df(df)

# Using OpenAI API to get the answer
response = openai.Completion.create(
  model=COMPLETIONS_MODEL,
  prompt=prompt,
  max_tokens=100
)
"""
openai.Completion.create(
    prompt=prompt,
    temperature=0,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    model=COMPLETIONS_MODEL
)["choices"][0]["text"].strip(" \n")
"""

# Print the response from OpenAI
print(response.choices[0].text.strip())

question = "Which Candidate has the most experience in payment platforms?"
prompt =  context + "\nQ: " + question + "\n" + "A: "

print(prompt)

