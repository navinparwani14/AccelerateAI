from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import tempfile
import warnings
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, base64
import asyncio
import edge_tts
from streamlit_mic_recorder import speech_to_text

warnings.filterwarnings('ignore')
load_dotenv()

# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
# os.environ['GOOGLE_API_KEY'] = os.getenv("GEMINI_API_KEY")     
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

voices = {
    "William":"en-AU-WilliamNeural",
    "James":"en-PH-JamesNeural",
    "Jenny":"en-US-JennyNeural",
    "US Guy":"en-US-GuyNeural",
    "Sawara":"hi-IN-SwaraNeural",
}

# Set page configuration with custom theme
st.set_page_config(
    page_title="Clinical Trials Chatbot", 
    layout="wide", 
    page_icon="🧪",
    initial_sidebar_state="expanded"
)


# Add CSS for the scrollable queries container only
st.markdown("""
    <style>
    .demo-queries-container {
        max-height: 200px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        margin-bottom: 10px;
        background-color: white;
    }
    </style>
""", unsafe_allow_html=True)

def create_vectorstore():
    """Create and save the vector store if it doesn't exist"""
    try:
        # Check if we can load the existing vector store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local("FAISS", embeddings, allow_dangerous_deserialization=True)
        st.success("Vector store loaded successfully.")
        return vectorstore
    except Exception as e:
        # If the vector store doesn't exist or there's an error loading it, create a new one
        st.warning("Vector store not found or could not be loaded. Creating a new one...")
        with st.spinner("Creating vector store. This might take a moment..."):
            loader = CSVLoader("sampled_5000.csv")
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
            chunks = splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            vectorstore.save_local("FAISS")
            st.success("Vector store created and saved successfully.")
            return vectorstore

# Title
st.markdown("""
    <h1 style='text-align: center;'>
        AccelerateAI Clinical & Regulatory Intelligence 
    </h1>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("image.jpg", width=500)
    st.markdown("## AccelerateAI LLM")
    st.write("This bot can answer questions related to clinical trials.")
    st.divider()
    st.subheader("Voice Settings")
    # voice_enabled = st.toggle("Enable Voice Response", key="voice_toggle")
    # if voice_enabled:
    #     voice_option = st.selectbox("Choose a voice:", options=list(voices.keys()), key="voice_response")
    # st.divider()



# Categorized Demo Queries
categorized_queries = {
    "General Info": [
        "What are the phases of clinical trials?",
        "How long does a typical clinical trial last?",
        "What is informed consent in clinical trials?"
    ],
    "Design & Methodology": [
        "What is the difference between Phase 2 and Phase 3 trials?",
        "What is a placebo-controlled trial?",
        "What are primary and secondary endpoints?"
    ],
    "Participant Criteria": [
        "How are participants selected for clinical trials?",
        "What are inclusion and exclusion criteria?",
        "How are adverse events reported in clinical trials?",
        "What is the FDA approval process after clinical trials?"
    ]
}



# Select category
selected_category = st.sidebar.selectbox("Select Category", list(categorized_queries.keys()))

# Demo queries section with scrollable container
st.sidebar.subheader("Sample Questions")
st.sidebar.write("Click on any question to use it:")

# Initialize session state for selected query
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""

# Scrollable container
st.sidebar.markdown('<div class="demo-queries-container">', unsafe_allow_html=True)
for query in categorized_queries[selected_category]:
    if st.sidebar.button(query, key=f"btn_{query}"):
        st.session_state.selected_query = query
st.sidebar.markdown('</div>', unsafe_allow_html=True)




# Load or create vectorstore
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = create_vectorstore()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today?"}
    ]

def format_docs(docs):
    return "\n\n".join(
        [f'Document {i+1}:\n{doc.page_content}\n'
         f'Source: {doc.metadata.get("source", "Unknown")}\n'
         f'Category: {doc.metadata.get("category", "Unknown")}\n'
         f'Instructor: {doc.metadata.get("instructor", "N/A")}\n-------------'
         for i, doc in enumerate(docs)]
    )

# Reset conversation
def reset_conversation():
    st.session_state.pop('chat_history')
    st.session_state['chat_history'] = [
        {"role":"assistant", "content":"Hey there! How can I assist you today about clinical trials?"}
    ]
    st.session_state.selected_query = ""


def rag_qa_chain(question, retriever, chat_history):
    # llm = ChatGroq(model="llama-3.1-8b-instant")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    output_parser = StrOutputParser()

    # System prompt to contextualize the question
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
                        which might reference context in the chat history, formulate a standalone question \
                        which can be understood without the chat history. Do NOT answer the question, \
                        just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    
    contextualize_q_chain = contextualize_q_prompt | llm | output_parser

    qa_system_prompt =  """

                You are a clinical trials expert trusted by researchers, clinicians, and regulatory professionals. Your role is to deliver accurate, high-impact responses based on the provided content and your expert knowledge. Follow these directives:

            - Provide clear, precise answers using the retrieved context where applicable.
            - When context is insufficient, draw confidently from your clinical expertise without referencing missing or limited data.
            - Avoid phrases like "based on the context," "provided text," or any mention of data scope.
            - Maintain a professional, authoritative tone while ensuring your answers are easy to read and understand.
            - Use technical language appropriately (e.g., NCT numbers, endpoints, phases, randomization methods), but always prioritize clarity and usability.
            - Organize complex answers with clear structure: bullet points, numbered lists, or short paragraphs when needed.
            - If unsure of the user’s intent, give useful general guidance and ask a precise follow-up question to clarify.
            -   If you don't get the correct answer from vector database try to answer from your own knowledge.

            {context}
                """
                    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # final_llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.5)
    final_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, api_key=os.getenv("OPENAI_API_KEY"))
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualize_q_chain | retriever | format_docs
        )
        | prompt
        | final_llm
        | output_parser
    )
    
    return rag_chain.stream({"question": question, "chat_history": chat_history})



# Generate the speech from text
async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        await communicate.save(temp_file.name)
        temp_file_path = temp_file.name
    return temp_file_path

# Get audio player
# def get_audio_player(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#         b64 = base64.b64encode(data).decode()
#         return f'<audio autoplay="true" src="data:audio/mp3;base64,{b64}">'
        
# # Text-to-Speech function which automatically plays the audio
# def generate_voice(text, voice):
#     text_to_speak = (text).translate(str.maketrans('', '', '#-*_😊👋😄😁🥳👍🤩😂😎')) # Removing special chars and emojis
#     with st.spinner("Generating voice response..."):
#         temp_file_path = asyncio.run(generate_speech(text_to_speak, voice)) 
#         audio_player_html = get_audio_player(temp_file_path)  # Create an audio player
#         st.markdown(audio_player_html, unsafe_allow_html=True)
#         os.unlink(temp_file_path)



# Dividing the main interface into two parts
col1, col2 = st.columns([1, 5])

# Displaying chat history
for message in st.session_state.chat_history:
    avatar = "assets/user.png" if message["role"] == "user" else "assets/assistant.png"
    with col2:
        st.chat_message(message["role"], avatar=avatar).write(message["content"])


# Handle voice or text input
with col1:
    st.button("Reset", use_container_width=True, on_click=reset_conversation)

    with st.spinner("Converting speech to text..."):
        text = speech_to_text(language="en", just_once=True, key="STT", use_container_width=True)


query = st.chat_input("Type your question")

# Use query from demo selection if available
if st.session_state.selected_query:
    query = st.session_state.selected_query
    st.session_state.selected_query = ""  # Clear after using it

# Generate the response
if text or query:
    user_query = text if text else query
    col2.chat_message("user", avatar="assets/user.png").write(user_query)
    
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Generate response
    with col2.chat_message("assistant", avatar="assets/assistant.png"):
        try:
            response = st.write_stream(rag_qa_chain(question=user_query,
                            retriever=st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 6}),
                            chat_history=st.session_state.chat_history))
        
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Generate voice response if enabled
            # if "voice_response" in st.session_state and st.session_state.voice_response:
            #     response_voice = st.session_state.voice_response
            #     generate_voice(response, voices[response_voice])
                
        except Exception as e:
            st.error(f"An internal error occurred. Please check your internet connection")
