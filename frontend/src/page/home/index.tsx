import React, { useEffect, useState } from 'react';
import InputText from "./components/InputText";
import OutputText from "./components/OutputText"
import Header from "../../components/Header";
import { fetchData } from './api';
//import Footer from "../../components/Footer";
import "../../assets/css/Home/index.css"
export default function Home() {

  let [prediction, setPrediction] = useState();
  let [questions, setQuestions] = useState<any[]>([]);

  function SetPrediction(x: any) {
    setPrediction(x);
    if (x && x.questions) {
      setQuestions(x.questions);
    } else {
      setQuestions([]);
    }
  }

  useEffect(() => {
    fetchData()
      .then((data) => {
        setPrediction(data);
        setQuestions(data.questions);
      })
      .catch((error) => {
        console.error(error);
      });
  }, []); 

  function outputText() {
    if (prediction != null) {
      return <OutputText questions={questions} setQuestions={setQuestions} />;
    } else {
      return <></>;
    }
  }
  
  return (
    <div className="body flex">
      <Header />
      <div className="body-main">
        <div className="flex items-center justify-center text-xl font-bold mb-4 text-black-500">
          Generate different quizzes like MCQs, True or False, Fill-in-the-blanks, FAQs, etc using AI
        </div>
        <div className="flex flex-row">
          <InputText prediction={prediction} SetPrediction={SetPrediction} />
          {outputText()}
        </div>
      </div>
    </div>
  )
}