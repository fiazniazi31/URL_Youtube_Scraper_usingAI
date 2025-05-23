import re
import validators
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader

# Streamlit setup
st.set_page_config(page_title="URL Summary", page_icon="📺")
st.title("📜 Text Summarizer")
st.sidebar.title("Summarize URL")

with st.sidebar:
    groq_api_key = st.text_input("GROQ API key", value="", type="password")

generic_url = st.text_input("URL : ", label_visibility="collapsed")

prompt_template = PromptTemplate(
    template="""
Provide a summary of the following content in 300 words:
Content:{text}
""",
    input_variables=["text"],
)

if st.button("Summarize"):
    # 1) Basic validation
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please input both the API key and a URL to get started.")
        st.stop()
    if not validators.url(generic_url):
        st.error("Please enter a valid URL (web page or YouTube video).")
        st.stop()

    # 2) Initialize LLM
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

    try:
        with st.spinner("Fetching & summarizing..."):
            docs = []
            if "youtube.com/watch" in generic_url:
                # Extract video ID
                m = re.search(r"v=([A-Za-z0-9_-]{11})", generic_url)
                if not m:
                    raise ValueError("Could not parse YouTube video ID.")
                video_id = m.group(1)

                try:
                    # Pull transcript
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                    text = " ".join([seg["text"] for seg in transcript])
                except TranscriptsDisabled:
                    st.error("Transcripts are disabled for this video.")
                    st.stop()

                docs = [Document(page_content=text)]

            else:
                # Fallback to URL loader
                loader = UnstructuredURLLoader(
                    urls=[generic_url],
                    ssl_verify=False,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/136.0.0.0 Safari/537.36"
                        )
                    },
                )
                docs = loader.load()

            # 3) Summarize chain
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt_template)
            summary = chain.run(docs)

            st.success(summary)

    except Exception as e:
        st.error(f"Exception: {e}")
