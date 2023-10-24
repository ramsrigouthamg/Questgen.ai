import axios from 'axios';
import fs from 'fs';

export async function predict(inputText: string): Promise<any> {

  let prediction = {
    "questions": [
      {
        "answer": "capital city",
        "context": "London, the capital city of the United Kingdom, stands as a testament to the intertwining of history, culture, and modernity.",
        "extra_options": [
          "Little Village"
        ],
        "id": 1,
        "options": [
          "Main City",
          "Capitol",
          "Northern Part"
        ],
        "options_algorithm": "sense2vec",
        "question_statement": "What is London?",
        "question_type": "MCQ"
      },
      {
        "answer": "intertwining",
        "context": "London, the capital city of the United Kingdom, stands as a testament to the intertwining of history, culture, and modernity.",
        "extra_options": [
          "Ties"
        ],
        "id": 2,
        "options": [
          "Intertwine",
          "Wove",
          "Deepening"
        ],
        "options_algorithm": "sense2vec",
        "question_statement": "What is the relationship between history, culture, and modernity?",
        "question_type": "MCQ"
      },
      {
        "answer": "england",
        "context": "Nestled along the banks of the River Thames in southeastern England, this metropolis has a storied past that stretches back over two millennia.",
        "extra_options": [
          "Italy",
          "Belgium",
          "Cornwall",
          "Denmark"
        ],
        "id": 3,
        "options": [
          "Ireland",
          "Wales",
          "Spain"
        ],
        "options_algorithm": "sense2vec",
        "question_statement": "What country is the metropolis in southeastern England?",
        "question_type": "MCQ"
      }
    ],
    "statement": "London, the capital city of the United Kingdom, stands as a testament to the intertwining of history, culture, and modernity. Nestled along the banks of the River Thames in southeastern England, this metropolis has a storied past that stretches back over two millennia. Its evolution from the Roman settlement of Londinium to the bustling global city it is today is a journey through time and significance.",
    "time_taken": 8.821425914764404
  }
  return prediction
  /*
  let data = { "input_text": inputText };

  let config = {
    method: 'get',
    maxBodyLength: Infinity,
    url: 'http://localhost:8000/predict',
    data: data
  };

  await axios.request(config)
  .then((response) => {
    console.log("test")
    console.log(response)
    return(JSON.stringify(response.data));
  })
  .catch((error: any) => {
    return(error);
  });
  */
};