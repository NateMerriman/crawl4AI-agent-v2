from dotenv import load_dotenv
import streamlit as st
import asyncio
import os
from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
)


# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter,
)

from rag_agent import agent, RAGDeps
from utils import get_chroma_client

load_dotenv()


async def get_agent_deps(
    collection_name_to_use: str, n_results_to_use: int
):  # <-- Add argument
    return RAGDeps(
        chroma_client=get_chroma_client("./chroma_db"),
        collection_name=collection_name_to_use,  # <-- Use argument
        embedding_model="all-MiniLM-L6-v2",  # Make sure this matches your model if you change it later
        n_results=n_results_to_use,  # <-- Store the value here
    )


def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # user-prompt
    if part.part_kind == "user-prompt":
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == "text":
        with st.chat_message("assistant"):
            st.markdown(part.content)


async def run_agent_with_streaming(
    user_input, n_results_to_retrieve: int, selected_temperature: float
):  # <-- Add temperature parameter
    async with agent.run_stream(
        user_input,
        deps=st.session_state.agent_deps,
        message_history=st.session_state.messages,
        # --- Ensure this line is present ---
        n_results=n_results_to_retrieve,
        temperature=selected_temperature,
        # --- End of line to check ---
    ) as result:
        async for message in result.stream_text(delta=True):
            yield message

    # Add the new messages to the chat history (including tool calls and responses)
    st.session_state.messages.extend(result.new_messages())


# --- Add the entire block below ---
DB_DIR = "./chroma_db"  # Or your configured DB directory
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Or your configured model

try:
    chroma_client = get_chroma_client(DB_DIR)
    available_collections = [col.name for col in chroma_client.list_collections()]
    if not available_collections:
        st.error(f"No collections found in {DB_DIR}. Please run insert_docs.py first.")
        st.stop()
except Exception as e:
    st.error(f"Failed to connect to ChromaDB at {DB_DIR}: {e}")
    st.stop()
# --- End of added block ---

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


async def main():
    st.title("ChromaDB Crawl4AI RAG AI Agent")

    # --- Add the dropdown right after the title ---
    selected_collection_name = st.selectbox(
        "Select Knowledge Base:", options=available_collections
    )
    # --- End of added dropdown ---

    # --- Add this line for n_results ---
    n_results_to_retrieve = st.number_input(
        "Number of Chunks to Retrieve:", min_value=1, max_value=20, value=5, step=1
    )
    # --- End of added line ---

    # --- Add this line for temperature ---
    selected_temperature = st.slider(
        "LLM Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.05
    )
    # --- End of added line ---

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Check if agent_deps needs to be created or updated based on selection
    # This logic ensures that if the user changes the dropdown, the agent deps are refreshed
    current_deps_collection = getattr(
        st.session_state.get("agent_deps"), "collection_name", None
    )
    current_deps_n_results = getattr(
        st.session_state.get("agent_deps"), "n_results", None
    )  # <-- Get current n_results
    if (
        "agent_deps" not in st.session_state
        or current_deps_collection != selected_collection_name
        or current_deps_n_results != n_results_to_retrieve  # <-- Add this check
    ):
        st.session_state.agent_deps = await get_agent_deps(
            selected_collection_name,
            n_results_to_retrieve,  # <-- Make sure this uses the variable from st.number_input
        )

        st.session_state.messages = []  # Optionally clear messages when collection changes
        st.rerun()  # Rerun to refresh the chat display after changing collection

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What do you want to know?")

    if user_input:
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Create a placeholder for the streaming text
            message_placeholder = st.empty()
            full_response = ""

            # Properly consume the async generator with async for
            generator = run_agent_with_streaming(
                user_input, n_results_to_retrieve, selected_temperature
            )  # <-- Pass temperature here
            async for message in generator:
                full_response += message
                message_placeholder.markdown(full_response + "▌")

            # Final response without the cursor
            message_placeholder.markdown(full_response)

            # --- Add this block to display sources ---
            try:
                # Get the ChromaDB collection object using the selected name
                collection = get_or_create_collection(
                    st.session_state.agent_deps.chroma_client,
                    st.session_state.agent_deps.collection_name,
                    embedding_model_name=st.session_state.agent_deps.embedding_model,
                )

                # Re-run the query using the user's input to find relevant sources
                # Re-run the query using the user's input to find relevant sources
                query_results = query_collection(
                    collection, user_input, n_results=n_results_to_retrieve
                )  # <-- Use the variable

                # Extract the source URLs from the metadata
                metadatas = query_results.get("metadatas", [[]])[0]
                # Get a unique list of valid source URLs
                source_urls = sorted(
                    list(
                        set(
                            [
                                meta.get("source")
                                for meta in metadatas
                                if meta and meta.get("source")
                            ]
                        )
                    )
                )

                # Display the sources in an expander
                if source_urls:
                    with st.expander("Retrieved Sources"):
                        for url in source_urls:
                            st.markdown(f"- {url}")  # Display each source URL
                # else: # Optionally handle case with no sources
                #    st.caption("No specific sources retrieved for this query.")

            except Exception as e:
                # Display error if source retrieval fails, but don't crash the app
                st.error(f"Error retrieving sources: {e}")
            # --- End of added block ---


if __name__ == "__main__":
    asyncio.run(main())
