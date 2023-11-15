# Questgen AI   <br>

## Try advanced question generation models for free:  https://questgen.ai/  


Questgen AI is an opensource NLP library focused on developing easy to use Question generation algorithms.<br>
It is on a quest build the world's most advanced question generation AI leveraging on state-of-the-art transformer models like T5, BERT and OpenAI GPT-2 etc.

## Online course and blog

 🚀 [Our online course that teaches how to build these models from scratch and deploy them](https://www.udemy.com/course/question-generation-using-natural-language-processing/?referralCode=C8EA86A28F5398CBF763)

[Blog announcing the launch](https://towardsdatascience.com/questgen-an-open-source-nlp-library-for-question-generation-algorithms-1e18067fcdc6)

<img src= './quest.gif' >

### Currently Supported Question Generation Capabilities :
<pre>
1. Multiple Choice Questions (MCQs)
2. Boolean Questions (Yes/No)
3. General FAQs
4. Paraphrasing any Question  
5. Question Answering.
</pre>

## Simple and Complete Google Colab Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1B_mYiUJziuyZygbxA1iP0RmM0iPP8qPB?usp=sharing)


## 1. Installation

### 1.1 Libraries
```
pip install git+https://github.com/ramsrigouthamg/Questgen.ai
pip install git+https://github.com/boudinfl/pke.git

python -m nltk.downloader universal_tagset
python -m spacy download en 
```
### 1.2 Download and extract zip of Sense2vec wordvectors that are used for generation of multiple choices.
```
wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
```

## 2. Running the code

### 2.1 Generate boolean (Yes/No) Questions
```
from pprint import pprint
import nltk
nltk.download('stopwords')
from Questgen import main
qe= main.BoolQGen()
payload = {
            "input_text": "Sachin Ramesh Tendulkar is a former international cricketer from India and a former captain of the Indian national team. He is widely regarded as one of the greatest batsmen in the history of cricket. He is the highest run scorer of all time in International cricket."
        }
output = qe.predict_boolq(payload)
pprint (output)
```

<details>
<summary>Show Output</summary>

```
'Boolean Questions': ['Is sachin ramesh tendulkar the highest run scorer in '
                       'cricket?',
                       'Is sachin ramesh tendulkar the highest run scorer in '
                       'cricket?',
                       'Is sachin tendulkar the highest run scorer in '
                       'cricket?']

```
</details>

### 2.2 Generate MCQ Questions
```
    qg = main.QGen()
    output = qg.predict_mcq(payload)
    pprint (output)
    
```

<details>
<summary>Show Output</summary>
            
```
    {'questions': [{'answer': 'cricketer',
                'context': 'Sachin Ramesh Tendulkar is a former international '
                           'cricketer from India and a former captain of the '
                           'Indian national team.',
                'extra_options': ['Mark Waugh',
                                  'Sharma',
                                  'Ricky Ponting',
                                  'Afridi',
                                  'Kohli',
                                  'Dhoni'],
                'id': 1,
                'options': ['Brett Lee', 'Footballer', 'International Cricket'],
                'options_algorithm': 'sense2vec',
                'question_statement': "What is Sachin Ramesh Tendulkar's "
                                      'career?',
                'question_type': 'MCQ'},
               {'answer': 'india',
                'context': 'Sachin Ramesh Tendulkar is a former international '
                           'cricketer from India and a former captain of the '
                           'Indian national team.',
                'extra_options': ['Pakistan',
                                  'South Korea',
                                  'Nepal',
                                  'Philippines',
                                  'Zimbabwe'],
                'id': 2,
                'options': ['Bangladesh', 'Indonesia', 'China'],
                'options_algorithm': 'sense2vec',
                'question_statement': 'Where is Sachin Ramesh Tendulkar from?',
                'question_type': 'MCQ'},
               {'answer': 'batsmen',
                'context': 'He is widely regarded as one of the greatest '
                           'batsmen in the history of cricket.',
                'extra_options': ['Ashwin', 'Dhoni', 'Afridi', 'Death Overs'],
                'id': 3,
                'options': ['Bowlers', 'Wickets', 'Mccullum'],
                'options_algorithm': 'sense2vec',
                'question_statement': 'What is the best cricketer?',
                'question_type': 'MCQ'}]}
```
</details> 


### 2.3 Generate FAQ Questions

```
output = qg.predict_shortq(payload)
pprint (output)
```


<details>
<summary>Show Output</summary>

 ```
 {'questions': [{'Answer': 'cricketer',
                'Question': "What is Sachin Ramesh Tendulkar's career?",
                'context': 'Sachin Ramesh Tendulkar is a former international '
                           'cricketer from India and a former captain of the '
                           'Indian national team.',
                'id': 1},
               {'Answer': 'india',
                'Question': 'Where is Sachin Ramesh Tendulkar from?',
                'context': 'Sachin Ramesh Tendulkar is a former international '
                           'cricketer from India and a former captain of the '
                           'Indian national team.',
                'id': 2},
               {'Answer': 'batsmen',
                'Question': 'What is the best cricketer?',
                'context': 'He is widely regarded as one of the greatest '
                           'batsmen in the history of cricket.',
                'id': 3}]
 }
 ```
</details>

### 2.4 Paraphrasing Questions
```
payload2 = {
    "input_text" : "What is Sachin Tendulkar profession?",
    "max_questions": 5
}
output = qg.paraphrase(payload2)
pprint (output)

```
<details>
<summary>Show Output</summary>
            
```
{'Paraphrased Questions': ["ParaphrasedTarget: What is Sachin Tendulkar's "
                           'profession?',
                           "ParaphrasedTarget: What is Sachin Tendulkar's "
                           'career?',
                           "ParaphrasedTarget: What is Sachin Tendulkar's job?",
                           'ParaphrasedTarget: What is Sachin Tendulkar?',
                           "ParaphrasedTarget: What is Sachin Tendulkar's "
                           'occupation?'],
 'Question': 'What is Sachin Tendulkar profession?'}
```
</details>

### 2.5 Question Answering (Simple)
```
answer = main.AnswerPredictor()
payload3 = {
    "input_text" : '''Sachin Ramesh Tendulkar is a former international cricketer from 
              India and a former captain of the Indian national team. He is widely regarded 
              as one of the greatest batsmen in the history of cricket. He is the highest
               run scorer of all time in International cricket.''',
    "input_question" : "Who is Sachin tendulkar ? "
    
}
output = answer.predict_answer(payload3)

```
<details>
<summary>Show Output</summary>
            
```
Sachin ramesh tendulkar is a former international cricketer from india and a former captain of the indian national team.
```
</details>

### 2.6 Question Answering (Boolean)
```
payload4 = {
    "input_text" : '''Sachin Ramesh Tendulkar is a former international cricketer from 
              India and a former captain of the Indian national team. He is widely regarded 
              as one of the greatest batsmen in the history of cricket. He is the highest
               run scorer of all time in International cricket.''',
    "input_question" : "Is Sachin tendulkar  a former cricketer? "
}
output = answer.predict_answer(payload4)
print (output)
```
<details>
<summary>Show Output</summary>
            
```
Yes, sachin tendulkar is a former cricketer.
```
</details>

### NLP models used

For maintaining meaningfulness in Questions, Questgen uses Three T5 models. One for Boolean Question generation, one for MCQs, FAQs, Paraphrasing and one for answer generation.

### Online Demo website.
https://questgen.ai/


[![Linkedin Link](linkedin.png)](https://www.linkedin.com/company/30182152/)
