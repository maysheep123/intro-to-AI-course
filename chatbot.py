import streamlit as st

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.messages import SystemMessage
from langchain_core.prompts import (HumanMessagePromptTemplate,
                                    ChatPromptTemplate)
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import (ChatOpenAI,
                            OpenAIEmbeddings)

from langchain_chroma import Chroma
from langchain_core.documents import Document

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
str_output_parser = StrOutputParser()

docs = [
    Document(page_content="Intro+to+AI+-+Course+notes"),
    #Document(page_content="Deep learning uses neural networks."),
    #Document(page_content="Python is commonly used for AI.")
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./intro-to-ai"
)

retriever = vectorstore.as_retriever(search_type = 'mmr',
                                     search_kwargs = {'k':1,
                                                      'lambda_mult':0.7})

openai_api_key = st.secrets["OPENAI_API_KEY"]

chat = ChatOpenAI(model_name='gpt-4o',
                 temperature=0,
                 openai_api_key=openai_api_key)

embedding = OpenAIEmbeddings(model='text-embedding-3-small')

PROMPT_S = '''YOU will receive a question from a student
taking the Intro to AI course.
Answer the question using only the provided context.'''

PROMPT_TEMPLATE_H = '''This is the question:
{question}

This is the context:
{context}'''

prompt_s = SystemMessage(PROMPT_S)
prompt_template_h = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE_H)

chat_prompt_template = ChatPromptTemplate([prompt_s,
                                          prompt_template_h])

chain = ({"context": retriever,
         "question": RunnablePassthrough()}
         | chat_prompt_template
         | chat
         | str_output_parser
        )

st.header("365 Q&A Chatbot", divider = True)

question = st.text_input("\nType your question:\n")

if st.button("Ask"):
    if question:
        respnose_placeholder = st.empty()
        response_text = ""

        result = chain.stream(question)

        for chunk in result:
            response_text += chunk
            respnose_placeholder.markdown(response_text)
    else:
        st.warning("Please type your question.", icon = "⚠️")