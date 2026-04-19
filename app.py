import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq
from pytubefix import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    CouldNotRetrieveTranscript,
)

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key is None:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file or environment.")

os.environ["GROQ_API_KEY"] = groq_api_key  

def get_youtube_transcript(url):
    try:
        video_id =YouTube(url).video_id
        api = YouTubeTranscriptApi()
        transcript_data = api.fetch(video_id, languages=['en'])
        text = " ".join([item.text for item in transcript_data])
        return text
    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable,CouldNotRetrieveTranscript) as e:
        st.error(f"Error fetching transcript: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
    
def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


st.title("Student Assistant RAG System")
st.write("Enter a YouTube video URL to fetch its transcript and create a QA system based on it.")

video_url = st.text_input("YouTube Video URL")

if st.button("Fetch Transcript and Answer Query"):
    if video_url:
        transcript_text = get_youtube_transcript(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            documents = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key), retriever=retriever)


            st.session_state.qa_chain = qa_chain
            st.success("Transcript fetched and QA chain created successfully!")

        else:
            st.warning("Could not fetch transcript. Please check the video URL and try again.")

if "qa_chain" in st.session_state:
    user_query = st.text_input("Ask a question about the video content:")
    if user_query:
        answer = st.session_state.qa_chain.run(user_query)
        st.write(f"**Answer**: {answer}")