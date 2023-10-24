import "../../../assets/css/Home/InputText.css"
import { predict } from "../../../service/home/predict"
import { useState } from 'react';

export default function InputText(props: any) {
    const [inputText, setInputText] = useState("Elon Musk has shown again he can influence the digital currency market with just his tweets. After saying that his electric vehicle-making company Tesla will not accept payments in Bitcoin because of environmental concerns, he tweeted that he was working with developers of Dogecoin to improve system transaction efficiency. \n \n Following the two distinct statements from him, the world's largest cryptocurrency hit a two-month low, while Dogecoin rallied by about 20 percent. The SpaceX CEO has in recent months often tweeted in support of Dogecoin, but rarely for Bitcoin.  In a recent tweet, Musk put out a statement from Tesla that it was concerned about the rapidly increasing use of fossil fuels for Bitcoin (price in India) mining and transaction, and hence was suspending vehicle purchases using the cryptocurrency. A day later he again tweeted saying, To be clear, I strongly believe in crypto, but it can't drive a massive increase in fossil fuel use, especially coal. It triggered a downward spiral for Bitcoin value but the cryptocurrency has stabilised since. \n\n A number of Twitter users welcomed Musk's statement. One of them said it's time people started realising that Dogecoin is here to stay and another referred to Musk's previous assertion that crypto could become the world's future currency."); 

    const handleInputChange = (e:any) => {
        setInputText(e.target.value); 
    }

    async function submit(){
        let prediction = await predict(inputText)
        props.SetPrediction(prediction) 
    }
    return (
        <div className="input-text text-md mb-2 flex-column flex">
            <div className="mb-2">
                Questions? Write to: hh11chelsea@gmail.com
            </div>
            <div className="mb-2">
                Supported text length: 5,000 words on the free plan. Supports English and other major languages.
                Note: Upgrade to generate quizzes from 25,000 words in one click.
            </div>
            <textarea className="h-[35vh] p-2 border rounded mb-2 text-md" onChange={handleInputChange} value={inputText}>
            </textarea>
            <select className="w-full p-2 mt-2 border rounded mb-2 hidden">
                <option value="MCQ">MCQ</option> 
                <option value="MCQ (Multiple Correct Answers)">MCQ (Multiple Correct Answers)</option>
                <option value="TrueFalse">TrueFalse</option>
                <option value="Fill in the blanks">Fill in the blanks</option>
                <option value="FAQ">FAQ</option>
                <option value="Higher Order QA">Higher Order QA</option>
            </select>
            <div className="flex mt-2 mb-2 hidden">
                <div className="w-1/2 pr-2">
                    <label id="numQuestions" className="block text-sm font-medium text-gray-700">Question Count</label>
                        <select id="numQuestions" name="numQuestions" className="w-full p-2 mt-1 border rounded">
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5</option>
                            <option value="6">6</option>
                            <option value="7">7</option>
                            <option value="8">8</option>
                            <option value="9">9</option>
                            <option value="10">10</option>
                            <option value="11">11</option>
                            <option value="12">12</option>
                            <option value="13">13</option>
                            <option value="14">14</option>
                            <option value="15">15</option>
                            <option value="16">16</option>
                            <option value="17">17</option>
                            <option value="18">18</option>
                            <option value="19">19</option>
                            <option value="20">20</option>
                        </select>
                </div>
                <div className="w-1/2 pl-2 hidden border-0-solid">
                    <label id="difficultyLevel" className="block text-sm font-medium text-gray-700">Difficulty Level</label>
                        <select id="difficultyLevel" name="difficultyLevel" className="w-full p-2 mt-1 border rounded">
                            <option value="Easy">Easy</option>
                            <option value="Medium">Medium</option>
                            <option value="Hard">Hard</option>
                        </select>
                </div>
            </div>
            <button className="w-full mt-2 p-2 bg-button-submit text-white rounded border-0-solid" onClick={() => submit()}>Submit</button>
        </div>
    )

}