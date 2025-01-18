import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.cohere import CohereEmbeddings
from typing import List, Tuple, Dict

# Set API Keys
os.environ["COHERE_API_KEY"] = "gij6DlDMOwcvn1KTlrahs34BbXiUJ88TbrF1vwOn"
os.environ["PINECONE_API_KEY"] = "pcsk_4FuV9h_HHo2LYZzYaNxk1LNdAE7yzDCodWiNtu2MppsoCGFX7W92mSzYWQXv3TXKSDLBs9"

class ChatHistory:
    """Manages chat history with proper context window management."""
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.messages: List[Dict[str, str]] = []
        self.conversation_pairs: List[Tuple[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to the history with proper formatting."""
        self.messages.append({"role": role, "content": content})
        
        # Update conversation pairs for the LLM context
        if role == "user":
            self._current_query = content
        elif role == "assistant" and hasattr(self, '_current_query'):
            self.conversation_pairs.append((self._current_query, content))
            self._maintain_history_window()
    
    def _maintain_history_window(self):
        """Maintain the conversation history within the specified window."""
        if len(self.conversation_pairs) > self.max_history:
            self.conversation_pairs = self.conversation_pairs[-self.max_history:]
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages for display."""
        return self.messages
    
    def get_conversation_pairs(self) -> List[Tuple[str, str]]:
        """Get conversation pairs for LLM context."""
        return self.conversation_pairs

def initialize_embeddings() -> CohereEmbeddings:
    """Initialize Cohere embeddings with improved configuration."""
    return CohereEmbeddings(
        model="embed-english-v3.0",
        cohere_api_key=os.environ["COHERE_API_KEY"],
        user_agent="app"
    )

def create_chat_prompt() -> ChatPromptTemplate:
    """Create a standardized AI response template with result format."""
    context = """You are an advanced AI assistant with access to a comprehensive Vector Database containing domain-specific knowledge. Your responsibilities include:

1. Accurately understanding user queries and context
2. Efficiently retrieving relevant data from the Vector Database
3. Delivering clear, precise, and well-structured answers
4. Ensuring the responses are accessible, concise, and informative

Response Guidelines:
- Always strive for accuracy by utilizing the Vector Database effectively
- Provide responses that are relevant to the user's specific query or need
- Use formatting to enhance clarity, ensuring ease of comprehension
- Ensure that each response is coherent and contextually aligned with the user's request

Documents:
{context}

Result Format:
- **Introduction:** A brief overview or summary addressing the user's query
- **Key Points:** A bulleted list or numbered points for clarity
- **Supporting Information:** Additional context, examples, or references where necessary
- **Conclusion:** A concise statement or recommendation if applicable"""

    system_template = SystemMessagePromptTemplate.from_template(context)
    human_template = HumanMessagePromptTemplate.from_template("{question}")
    
    return ChatPromptTemplate.from_messages([system_template, human_template])

def initialize_retrieval_chain() -> ConversationalRetrievalChain:
    """Initialize the conversational retrieval chain with proper configuration."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyDrWCigwixHASe-9C4fLUGsjz3OlhDzyqM",
        temperature=0.3,
        max_tokens=1524,
        timeout=45,
        max_retries=3,
    )
    
    embeddings = initialize_embeddings()
    retriever = PineconeVectorStore.from_existing_index(
        "new",
        embeddings
    ).as_retriever(
        search_kwargs={"k": 5}
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": create_chat_prompt(), "document_variable_name": "context"},
        chain_type="stuff",
        return_source_documents=False,
        verbose=True
    )

def run_query(qa_chain: ConversationalRetrievalChain, query: str, chat_history: List[Tuple[str, str]]) -> str:
    """Run a query through the QA chain with error handling."""
    try:
        result = qa_chain({
            "question": query, 
            "chat_history": chat_history
        })
        return result["answer"]
    except Exception as e:
        raise Exception(f"Error processing query: {str(e)}")

def main():
    # Page Configuration
    st.set_page_config(
        page_title="AI Assistant",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.sidebar.title("🤖 Welcome to Your AI Assistant")
    st.sidebar.markdown("""
        ### What I Can Do:
        - 🧠 **Retrieve Domain-Specific Knowledge**  
        Ask me about any topic, and I’ll provide detailed, relevant information.
        - ✅ **Provide Accurate & Concise Answers**  
        Get quick, to-the-point answers to your queries.
        - 🗂️ **Offer Well-Structured Insights**  
        Receive organized, insightful responses to complex questions.
        """)
    
    
    st.sidebar.info("""
    ### **How to Use Me:**
    1. **Type your question** in the chat box below.
    2. **Be specific** for better answers.
    3. Explore and learn—I'm here to assist!
    """)

    # Main Content
    st.title("💬 Your AI Assistant")
    st.markdown("""
    Welcome to your personal AI Assistant! I'm here to assist you with:  
    - 📘 Answering questions  
    - 💡 Providing detailed explanations  
    - 🔍 Offering suggestions and recommendations  
    - 🎯 Helping with a wide range of topics  

    ### **Guidelines for Best Experience:**
    - Be clear and precise in your queries.  
    - If you're exploring a new domain, provide some context.  
    - Let me know if you'd like a summary, detailed insights, or actionable recommendations.  

    ### **Get Started Now:**
    Simply type your question or topic of interest in the chat box below!
    """)

    # Initialize session state
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatHistory(max_history=5)

    # Initialize chain if not already initialized
    if st.session_state.qa_chain is None:
        with st.spinner("Initializing AI assistant..."):
            try:
                st.session_state.qa_chain = initialize_retrieval_chain()
                st.success("Ready to help!")
            except Exception as e:
                st.error(f"Error initializing system: {e}")
                return

    # Display chat messages
    for message in st.session_state.chat_history.get_messages():
        with st.chat_message(message["role"], avatar="💬" if message["role"] == "assistant" else None):
            st.write(message["content"])

    # Chat input
    query = st.chat_input("Ask me anything...")
    
    if query:
        # Add user message
        st.session_state.chat_history.add_message("user", query)
        with st.chat_message("user"):
            st.write(query)
        
        # Generate and display response
        with st.chat_message("assistant", avatar="💬"):
            with st.spinner("Thinking..."):
                try:
                    response = run_query(
                        st.session_state.qa_chain,
                        query,
                        st.session_state.chat_history.get_conversation_pairs()
                    )
                    st.write(response)
                    st.session_state.chat_history.add_message("assistant", response)

                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {e}. Please try asking your question differently."
                    st.error(error_message)
                    st.session_state.chat_history.add_message("assistant", error_message)

if __name__ == "__main__":
    main()