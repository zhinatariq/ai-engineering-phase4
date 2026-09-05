from fastapi import FastAPI
from pydantic import BaseModel
from graph import app as graph_app


api = FastAPI()


class AskRequest(BaseModel):
    question: str
    messages: list[dict]


@api.post("/ask")
def ask_question(request: AskRequest):
    result = graph_app.invoke({
        "question": request.question,
        "messages": request.messages,
        "answer": "",
    })

    return {"answer": result["answer"]}