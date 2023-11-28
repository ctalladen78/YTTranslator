import streamlit as st
from multiprocessing import Process
from streamlit_modal import Modal
import os
import signal
import requests
import pandas as pd
import youtube_transcript_api
from deep_translator import GoogleTranslator
import openai
import langchain
# from transformers import pipeline
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents import load_tools

from datetime import datetime, timedelta
import time

#Title
st.title("Youtube CJK translator")

language = ['ja','ko'zh-CN']
selected_lang = language[0]
# language = [{'ja':"🇯🇵"},{'ko':"🇰🇷"},{'zh-CN':"🇨🇳"}]
# selected_lang = [lang for lang in language if 'ja' in lang][0]['ja']


# modal = Modal("Demo Modal",
#     key="demo-modal",
#     # Optional
#     padding=20,    # default value
#     max_width=744  # default value
#     )
# close_modal = None
# if close_modal:
#     # modal.close()
#     os.kill(st.session_state.pid, signal.SIGKILL)
# PROC = Process(target=modal)

with st.sidebar:
    st.title('🍁 Translator App')
    st.markdown('''
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models)

    Inspired by : [YT-TLDR](https://www.you-tldr.com)
    ''')
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    #     radio button to select japanese by default
    st.radio("Language selected:",
            ["Japanese","Chinese","Korean"])
    st.write("### About")
    st.markdown('''
            - Personalized educational assistant
            - Practical learning through translation
            - Word analysis
        ''')

home_tab, summary_tab, qna_tab = st.tabs(["1️⃣ Translation", "2️⃣ Summary & Analysis", "3️⃣ Question & Answer"])

embeddings = None
llm = None
vectordb = None
summarizer = None
summary_text = ""
combined_text = ""
combined_text_ls = []
chat_history = []
combined_raw = ""

@st.cache_data
def get_timestamps(row):  
    start_time = datetime(1,1,1) + timedelta(seconds=row['start'])
    start_time_srt = start_time.strftime("%H:%M:%S,{:03d}".format(int(start_time.microsecond / 1000)))

    return start_time_srt


@st.cache_resource
def init_model():
    # pipeline('summarization',
        # model="t5-small",
        # model_kwargs={"cache_dir": './models'}
    # )
    # Initialize the OpenAI module, load and run the summarize chain
    llm=OpenAI(temperature=0, openai_api_key=openai_api_key)
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

    return 

    
# @st.cache(allow_output_mutation=True)
@st.cache_data
def get_translation(yt_path):
    # translate japanese
    vid = yt_path.split("=")[1]

    transcript_list = youtube_transcript_api.YouTubeTranscriptApi.list_transcripts(vid)

    transcript = transcript_list.find_transcript([f"{selected_lang}"])
    translated_text_ls = []
    try:
        transcript_fetched = transcript.fetch()
        transcript_text = [item['text'] for item in transcript_fetched]
        # combine_text(transcript_text)
        translator = GoogleTranslator(source=f"{selected_lang}", target='en')

        # for item in transcript_fetched
        for idx, item in enumerate(transcript_fetched):
            tr = translator.translate(str(item['text']))
            translated_text_ls.append({'start':float(item['start']),
                                   'duration':float(item['duration']),
                                   'text':str(item['text']),
                                   'translated':tr})
            print(translated_text_ls[idx])
        
    except:
        translated_text_ls = []
    return translated_text_ls

@st.cache_data
# # summarize all translated transcript content
def get_summarization(txt):
    llm=OpenAI(temperature=0, openai_api_key=openai_api_key)
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    chunks = text_splitter.split_text(txt)
    vectordb = FAISS.from_texts(chunks, embeddings)

    chain = load_summarize_chain(llm, chain_type="stuff")
    search = vectordb.similarity_search(" ")
    summary = chain.run(input_documents=search, question="Write a summary with at least 300 up to 700 words.")
    # summary = chain.run(input_documents=chunks, question="Write a summary within 1000 words.")

    return summary

@st.cache_data
def get_chat_response(prompt,txt):
    # langchain qna chain
    # https://github.com/hwchase17/chat-your-data/blob/master/query_data.py
    cllm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    chunks = text_splitter.split_text(summary_text)
    vectordb = FAISS.from_texts(chunks, embeddings)
    retriever = vectordb.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=cllm,
                                     chain_type="stuff", 
                                     retriever=retriever, 
                                     verbose=True)
    
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history", return_messages=True)
    # chain = load_qa_chain(llm=OpenAI(), chain_type="map_reduce")  
    # qa = ConversationalRetrievalChain.from_llm(llm=cllm, 
        # retriever=retriever,
        # memory=memory
        # )

    template = f"""
        You are a language expert assistant in Japanese, Chinese, Korean.
        If you are asked about unrelated expertise, then politely reply I don't know.
        You are expert in vocabulary, grammatical cases, conjugation, slang.
        Answer the question based on the following context:
        {summary_text}
        Question: {prompt}
    """
    # prompt = ChatPromptTemplate.from_template(template)
    # result = qa({"question": prompt, "chat_history": chat_history})
    result = qa.run(template)
    print(result)
    
    return result

@st.cache_data
def add_top_row(df):
    # Create a new row with initial values
    # new_row = pd.DataFrame({'Unnamed: 0': [0],
    new_row = pd.DataFrame({'start': [0.0],
                            'duration':df['start'][0],
                            'text':["🍁"],
                            'translated':["🍂"],
                            'timestamp':["00:00:00,000"]
                        },index=[0])
    # new_row.reset_index()
    df = pd.concat([new_row, df], ignore_index=True)
    return df

def print_data_one_by_one(row):
    # print(row['text'])
    # update subtitles text
    st.session_state['subtitle'] = row['text']
    # st.write(st.session_state['subtitle'])
    st.write(f"{row['start']} : {row['text']}")
    time.sleep(float(row['duration']))

@st.cache_data
def get_analysis(df):
    # tools = load_tools(["python_repl"])

    df_agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        df,
        verbose=True,
        # extra_tools=tools
        )
    context = """
    For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of chart, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The title with the highest rating is 'Gilead'"}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Return all output as a string.

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
    """
    res1 = df_agent.run(f"{context}. Show a bar chart by Top 10 words in {selected_lang}")
    st.write("Show a bar chart by Top 10 words in {selected_lang}")
    st.success(res1)
    # res2 = df_agent.run(f"{context} What is the most occurring {selected_lang} word?")
    # st.success(res2)
    # res3 = df_agent.run(f"{context} List the top most occuring {selected_lang} classified as nouns")
    # st.success(res1)


def main():
    # with modal.container():
    #     PROC.start()
    #     st.session_state.pid = PROC.pid
    #     close_modal = st.button("X") 

    # home
    with home_tab:
        URL = "https://www.youtube.com/watch?v=FiLHU4QiUs8"
        yt_path = st.text_input("Enter youtube link to translate...", 
            URL)
            # placeholder="https://www.youtube.com/watch?v=FiLHU4QiUs8")

        st.write("### YT Translation")
        if not openai_api_key:
            # st.error(f"Please provide the missing fields.")
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        elif yt_path is None:
            st.info("Please provide the missing fields.")
            st.stop()

        else:
            with st.spinner("Translating...."):
                # TODO implement progress bar
                translated_text_ls=get_translation(yt_path)
                translated_text_df = pd.DataFrame(translated_text_ls)
                # translated_text_df = translated_text_df['duration'].astype('float64')
                # translated_text_df = translated_text_df['start'].astype('float64')
                # add to new timestamps column
                translated_text_df['timestamp'] = translated_text_df.apply(get_timestamps,axis=1)
                translated_text_df = add_top_row(translated_text_df)
                combined_text_ls = translated_text_df.translated.tolist()
                combined_text = ' '.join(translated_text_df.translated)
                combined_raw = ' '.join(translated_text_df.text)

                st.success(f"Language detected: {selected_lang}")

                # TODO show pytube video
                st.video(URL)

                # show df
                st.dataframe(translated_text_df)


    # summary_tab
    with summary_tab:
        st.write("### Context Summary")
        if yt_path is None or openai_api_key is None:
            # st.error(f"Please provide the missing fields.")
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        else:
            with st.spinner("Summarizing...."):
                print(combined_text)
                summary_txt = get_summarization(combined_raw)
                st.success(summary_txt)
                # get_analysis(translated_text_df)

    # qna_tab
    with qna_tab:
        # 80-20 rule
        # TODO most occurring japanese words w translation
        st.write("### Question & Answer")
        if yt_path is None or openai_api_key is None:
            # st.error(f"Please provide the missing fields.")
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        else:
            # get_chat response
            prompt = st.text_input("Chat using summary",
                "お席をお見受けいたしましたら、お声をおかけいたします"
            )
            if prompt is not None:
                res = get_chat_response(prompt,combined_raw)
                chat_history.append({"prompt":"res"})
                print(chat_history)
                st.success(res)

    #     if prompt := st.chat_input():
    #         if not openai_api_key:
    #             st.info("Please add your OpenAI API key to continue.")
    #             st.stop()

    #         # prompt template
    #         openai.api_key = openai_api_key
    #         st.session_state.messages.append({"role": "user", "content": prompt})
    #         st.chat_message("user").write(prompt)
    #         response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    # with about_tab:
        

if __name__ == "__main__":
    main()