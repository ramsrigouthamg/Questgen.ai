import { useState } from "react";
import "../../../assets/css/Home/OutputText.css"


export default function OutputText(props: any) {
    let [isEdit, setIsEdit] = useState(false);

    function questionStatementChange(questionId: any, questionStatement: any) {
        let questions = props.questions
        for (let q of questions) {
            if (questionId == q.id) {
                q.question_statement = questionStatement;
                console.log(questions)
            }
        }
        props.setQuestions(questions)
    }

    function Question(question: any) {
        if (!isEdit)
            return (
                <div className="mb-6 bg-white shadow-lg rounded p-6 relative flex flex-col space-y-4">
                    <div className="absolute top-0 left-0 bg-gray-50 p-1 rounded text-sm text-gray-600 m-1">{question.id}/{props.questions.length}</div>
                    <h5 className="font-semibold text-lg mb-4 ">{question.question_statement}</h5>
                    <ul className="list-inside space-y-2">
                        {question.options.map((option: any) => (
                            <div className="grid grid-cols-12 items-center gap-4 mt-2">
                                <div className="col-span-1 flex justify-center items-center">
                                </div>
                                <div className="col-span-11 items-center text-md">
                                    {option}
                                </div>
                            </div>
                        ))}
                    </ul>
                </div>
            )
        else return (
            <div className="mb-6 bg-white shadow-lg rounded p-6 relative flex flex-col space-y-4">
                <div className="absolute top-0 left-0 bg-gray-50 p-1 rounded text-sm text-gray-600 m-1">{question.id}/{props.questions.length}</div>
                <input className="font-semibold text-lg mb-4 border-2 border-gray-300 p-2 rounded-md" type="text" value={question.question_statement}></input>
                <ul className="list-inside space-y-2">
                    {question.options.map((option: any) => (
                        <div className="grid grid-cols-12 items-center gap-4 mt-2">
                            <div className="col-span-1 flex justify-center items-center">
                                <input className="ml-2" type="radio" />
                            </div>
                            <input className="col-span-11 items-center border-2 border-gray-300 p-2 rounded-md text-md" value={option} />
                        </div>
                    ))}
                </ul>
                <div className="self-end cursor-pointer">
                </div>
            </div>
        )
    }

    if (props.questions != null)
        return (
            <div className="w-1/2 pl-2 relative">
                <div className="flex flex-wrap justify-end mb-4">
                    <div className="flex items-center space-x-2">
                        <label>Show Answers</label>
                    </div>
                    {isEdit ? (
                        <button className="button-edit-true" onClick={() => setIsEdit(false)}>
                            Edit
                        </button>
                    ) : (
                        <button className="button-edit" onClick={() => setIsEdit(true)}>
                            Edit
                        </button>
                    )}

                    <button className="button-edit">
                        Export
                    </button>
                </div>
                <form>
                    {props.questions.map((question: any) => Question(question))}
                </form>
            </div>
        )
    else return (<></>)
}