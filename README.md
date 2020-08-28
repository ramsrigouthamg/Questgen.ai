# Questgen AI    <br>
https://questgen.ai/  


Questgen AI is focused on developing Question generation algorithms using state-of-the-art Natural Language Processing techniques. It is very focused on ease of use. 

### Currently Supported Question Generation Capabilities :
<pre>
1. Multiple Choice Questions (MCQs)
2. Boolean Questions (Yes/No)
3. General FAQs
4. Paraphrasing any Question  
5. Question Answering.
</pre>

<img src= './quest.gif' >

### Simple and Complete Google Colab Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CvgSjU48kN5jEtCU732soM723W1spGdm?usp=sharing)



### Prerequisites
```
pip install -r requirements.txt
pip install git+https://github.com/boudinfl/pke.git
python -m nltk.downloader universal_tagset
python -m spacy download en 
```
### Using the library
```
pip install git+https://github.com/ramsrigouthamg/Questgen.ai
wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf  s2v_reddit_2015_md.tar.gz
```

**For MCQ, Short Question and Paraphrasing Question generation**
```
import Questgen
generator= main.QGen()                          #instance of QGen class

payload={
    "input_text" :   'Text',
    "max_questions" : 5                         #Default 4
    }
    
output1= generator.predict_mcq(payload)         #For MCQ generation       
output2= generator.predict_shortq(payload)      #For Short Answers' Question generaiton
output3= generator.paraphrase(payload)          #For paraphrasing questions
```


**For Boolean question generation**
```
import Questgen
generator= main.BoolQGen()                      #instance of BoolQGen class

payload={
    "input_text" :   'Text',
    "max_questions" : 5                         #Default 4
    }

output= generator.predict_boolq(payload)
```


**For Answer prediction from a given question**
```
import Questgen
generator= main.AnswerPredictor()

payload={
    "input_text" :   'Text',
    "input_question" : 'Question'                         
    }

output= generator.predict_answer(payload)
```



### NLP models used

For maintaining meaningfulness in Questions, Questgen uses Three T5 models, one for Boolean Question generation, one for MCQ, Short Questions, Paraphrasing and one for Answer generation.

### Online Demo website
Under development...
https://questgen.ai/
[![Linkedin Link](linkedin.png)](https://www.linkedin.com/company/30182152/)
