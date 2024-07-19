import asyncio
from typing import AsyncIterable
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from pydantic import BaseModel
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
import os
import json

# langchain의 기록을 확인할 수 있습니다.
import langchain
langchain.debug = True

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

os.environ["OPENAI_API_KEY"] = openai_api_key

embeddings = OpenAIEmbeddings()

# DB 불러오기 및 검색기 생성
vectorstore = FAISS.load_local('./db/faiss', embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# API 서버 관련 셋팅
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    content: str

async def send_message(content: str) -> AsyncIterable[str]:
    # content 파싱
    data = json.loads(content)
    occupation = data['occupation']
    gender = data['gender']
    experience = data['experience']
    cover_letter = data['cover_letter']

    # DB에서 관련 데이터 필터링
    relevant_docs = await retriever.aget_relevant_documents(f"{occupation} {gender} {experience}")
    context = "\n".join([doc.page_content for doc in relevant_docs])

    callback = AsyncIteratorCallbackHandler()
    # llm = ChatOllama(model="EEVE-Korean-10.8B:latest", streaming=True, verbose=True, callbacks=[callback],)
    llm = ChatOllama(model="gemma2:latest", streaming=True, verbose=True, callbacks=[callback],)

    prompt = ChatPromptTemplate.from_template(f"""
    You are an interview assistant bot. You must answer in Korean.
    Your job is to find the most unique 3 similar interview case from the database based on the provided cover letter.
    Do not modify the data retrieved from the database in any way.
    #Context: 
    {context} 
    Always respond in the following Key and Value. Must return in JSON format:
    "title": "채용면접 데이터에 기반한 유사 인터뷰 사례",
    "Question 1": "[Question 1]",
    "Answer 1": "[Answer 1]",
    "Question 2": "[Question 2]",
    "Answer 2": "[Answer 2]",
    "Question 3": "[Question 3]",
    "Answer 3": "[Answer 3]"
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = await chain.ainvoke(content)
    return response

# API 서버 생성
@app.post("/stream_chat/")
async def stream_chat(message: Message):
    response = await send_message(message.content)

    # # 이놈이 마크다운 형식으로 해준다면 지워버리기
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    print("\n\n\n\n🦖🦖🦖🦖🦖🦖Response received from LLM🦖🦖🦖🦖🦖🦖\n\n\n\n", response)
    try:
        response_json = json.loads(response)
        return JSONResponse(content=response_json)
    except json.JSONDecodeError as e:
        print("JSONDecodeError:", e)
        return JSONResponse(content={"error": "Failed to decode JSON response", "response": response}, status_code=500)


# api서버 띄우기 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)