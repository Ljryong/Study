# langchain version = 0.2.2

import streamlit as st
# from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
# from langchain.schema import RunnableSequence
# OpenAI API 키 설정
import os

# GPT-3 모델 초기화
llm = ChatOpenAI(temperature=0.5,               # 창의성 (0.0 ~ 2.0) 
                 max_tokens=2048,             # 최대 토큰수
                 model_name='gpt-4',  # 모델명
                 openai_api_key='' 
                )

<<<<<<<<< Temporary merge branch 1

=========
>>>>>>>>> Temporary merge branch 2
# 대화 메모리 설정
memory = ConversationBufferMemory(memory_key="history")

# 프롬프트 템플릿 설정
template = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are MindGuide, a compassionate and experienced mental health therapist. "
        "Here is the conversation history:\n{history}\n\n"
        "User: {input}\n"
        "MindGuide:"
    )
)

# LLMChain 설정
chain = LLMChain(
    llm=llm,
    prompt=template,
    memory=memory
)

# Streamlit UI 설정
st.title("MindGuide: Mental Health Support Chatbot")
st.write("안녕하세요! MindGuide 챗봇에 오신 것을 환영합니다. 정신 건강 문제나 고민을 자유롭게 나누어 주세요. 이곳은 안전하고 비밀이 보장되는 공간입니다.")

# 사용자 입력 처리
user_input = st.text_input("당신의 고민을 입력하세요:")

if user_input:
    # 사용자 입력을 통해 응답 생성
    response = chain.run({"input": user_input})
    
    # 응답 출력
    st.write("MindGuide:")
    st.write(response)

    # 이전 대화 내용 갱신 (메모리 수동 업데이트)
    # memory.chat_memory.messages.append(HumanMessage(content=user_input))
    # memory.chat_memory.messages.append(AIMessage(content=response))

# 이전 대화 내용 표시
st.write("대화 기록:")
for message in memory.chat_memory.messages:
    role = "User" if isinstance(message, HumanMessage) else "MindGuide"
    st.write(f"{role}: {message.content}")