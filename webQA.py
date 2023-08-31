# pip install pycryptodome
from glob import glob
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import requests
from bs4 import BeautifulSoup

#Qdrantというベクトルデータベースのローカルパス
QDRANT_PATH = "./local_qdrant"
#Qdrant内で使用するコレクション（データセット）の名前を定義
COLLECTION_NAME = "my_collection_2"

#初期ページの設定
def init_page():
    st.set_page_config(
        page_title="Ask",
        page_icon="?"
    )
    #サイドバーの名前
    st.sidebar.title("Nav")
    st.session_state.costs = []

#モデルを選択する関数
def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k"
    else:
        st.session_state.model_name = "gpt-4"
    #選択されたモデルのコンテキストサイズ（最大トークン数）をセッション状態に保存
    st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)

#Qdrantデータベースに接続するための関数
def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


# def build_vector_store(pdf_text):
#     qdrant = load_qdrant()
#     qdrant.add_texts(pdf_text)

#質問応答（QA）モデルを構築するための関数
def build_qa_model(llm):
    qdrant = load_qdrant()
    #Qdrantオブジェクトからretriever（情報検索器）を作成
    # 検索タイプは"similarity"（類似性）で、上位10件の結果を返すよう設定
    retriever = qdrant.as_retriever(
        search_type="similarity",
        search_kwargs={"k":10}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

#質問応答（QA）モデルを用いて質問に回答を生成するための関数
def ask(qa, query):
    with get_openai_callback() as cb:
        answer = qa(query)

    return answer, cb.total_cost

#askページの設定
def page_ask():
    st.title("Ask")

    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost = ask(qa, query)
                st.session_state.costs.append(cost)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


#URLからテキストデータの取得を行う関数
def get_article_text(url):
    # URLからHTMLを取得
    r = requests.get(url)
    r.raise_for_status()
    
    # HTMLからテキストを抽出
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    
    # テキストをトークン化
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-ada-002",
        chunk_size=500,
        chunk_overlap=0,
    )
    
    return text_splitter.split_text(text)

#URLアップロードページの設定
def page_url_upload_and_build_vector_db():
    st.title("URL Upload")
    container = st.container()
    with container:
        url = st.text_input("Input URL of the article:")
        if url:
            with st.spinner("Fetching and parsing article ..."):
                article_text = get_article_text(url)
                build_vector_store(article_text)

#メイン関数
def main():
    init_page()
    #サイドバーでページを選択
    selection = st.sidebar.radio("Go to", ["URL Upload", "Ask My Texts"])   
    if selection == "URL Upload":
        page_url_upload_and_build_vector_db()

    elif selection == "Ask My Texts":
        page_ask() 

    #コストを表示
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()