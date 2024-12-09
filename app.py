import streamlit as st
from chatbot.chatbot import Chatbot
from chatbot.embeddings import clone_repo, get_docs
from chatbot.pinecone_utils import insert_vectors, search_vectors, connect_pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_KEY

# List of available language models
available_models = ["gpt-4"]

# Set up the Streamlit app
st.title("cobase")

# Initialize the chatbot in session state if not already present
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

# Initialize the repo_url in session state if not already present
if "repo_url" not in st.session_state:
    st.session_state.repo_url = ""

# Sidebar for GitHub repository URL and model selection
with st.sidebar:
    st.subheader("Configuration")
    st.session_state.repo_url = st.text_input(
        "Enter GitHub Repository URL", st.session_state.repo_url
    )

    if st.session_state.repo_url:
        print(f"Selected repository: {st.session_state.repo_url}")

    # Dropdown for selecting the language model
    selected_model = st.selectbox("Select Language Model", available_models)
    with st.expander("ℹ️ Information about the selected model"):
        st.info(f"You have selected **{selected_model}**.")
        st.write(
            "Currently we are only compatible with ChatGPT 40 mini. We plan to add more LLM's in future."
        )

    if st.button("Create/Load Embeddings"):
        if st.session_state.repo_url:
            with st.spinner("Creating/Loading embeddings..."):
                try:
                    # Extract the repo name from the URL
                    codebase_name = st.session_state.repo_url.split("/")[-1].replace(
                        ".git", ""
                    )
                    print(f"Selected repo is: {codebase_name}")
                    # Clone the repository
                    repo_dir = clone_repo(st.session_state.repo_url, codebase_name)

                    # Connect to Pinecone
                    connect_pinecone()
                    print("Connection to Pinecone successful")

                    # Generate embeddings and insert into Pinecone
                    docs = get_docs(codebase_name)  # Load documents from the repo
                    print(f"Loaded documents: {len(docs)} documents")
                    for doc in docs[:5]:  # Print details of the first 5 documents
                        print(
                            f"Doc Metadata: {doc.metadata}, Content Preview: {doc.page_content[:100]}"
                        )
                    if docs:
                        embeddings = HuggingFaceEmbeddings(
                            model_name="all-MiniLM-L6-v2"
                        )
                        print("Embeddings initialized")
                        vectors = [
                            embeddings.embed_query(doc.page_content) for doc in docs
                        ]
                        print(f"Generated vectors: {len(vectors)} vectors")
                        metadata = [
                            {
                                "source": doc.metadata.get("source", "Unknown"),
                                "content": doc.page_content[:2000],
                            }
                            for doc in docs
                        ]
                        print("Metadata prepared")
                        insert_vectors(vectors, metadata, namespace=codebase_name)
                        st.success(
                            f"Embeddings for {codebase_name} created/loaded successfully in Pinecone!"
                        )
                    else:
                        st.warning(
                            f"No documents found in the repository {codebase_name}."
                        )

                    # Initialize the chatbot with the selected codebase and language model
                    st.session_state.chatbot = Chatbot(
                        model=selected_model, codebase_name=codebase_name
                    )
                except Exception as e:
                    st.error(f"Error creating/loading embeddings: {e}")
        else:
            st.warning("Please enter a GitHub repository URL.")


# Function to stream the response in chunks
def stream_response(response_text):
    message_placeholder = st.empty()
    full_message = ""

    for i in range(0, len(response_text), 50):  # Stream in chunks of 50 characters
        chunk = response_text[i : i + 50]
        full_message += chunk
        message_placeholder.markdown(
            full_message + "▌"
        )  # Add a cursor to indicate typing
        time.sleep(0.1)  # Adjust the delay to make it look natural

    message_placeholder.markdown(full_message)  # Final message without cursor


# Tab 1: Chatting with Codebase
tabs = st.tabs(["Chat with Codebase"])

with tabs[0]:
    st.write("Chat with your Codebase")

    if question := st.chat_input("Ask your codebase..."):
        if st.session_state.repo_url and selected_model:
            try:
                if st.session_state.chatbot is None:
                    codebase_name = st.session_state.repo_url.split("/")[-1].replace(
                        ".git", ""
                    )
                    st.session_state.chatbot = Chatbot(
                        model=selected_model, codebase_name=codebase_name
                    )
                    print(f"Chatbot initialized for repo: {codebase_name}")

                # Query the index
                query_vector = st.session_state.chatbot.embeddings.embed_query(question)
                print(f"Query vector generated: {query_vector[:5]} (truncated)")

                search_results = search_vectors(
                    query_vector, top_k=10, namespace=st.session_state.repo_url
                )
                print(f"Search results: {search_results}")

                # Display related documents
                matches = search_results.get("matches", [])
                if matches:
                    st.write("### Related Documents")
                    for match in matches:
                        doc_metadata = match.get("metadata", {})
                        source = doc_metadata.get("source", "Unknown source")
                        score = match.get("score", "No score")
                        st.write(f"- Source: {source}, Score: {score}")
                else:
                    st.warning("No related documents found.")

                # Generate and display the response
                st.write("### Question")
                stream_response(question)

                st.write("### Answer")
                response_content = st.session_state.chatbot.get_response(question)
                if response_content:
                    stream_response(response_content["answer"])
                else:
                    st.warning("The model did not return a response. Please try again.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning(
                "Please enter a question and configure the settings in the sidebar."
            )
