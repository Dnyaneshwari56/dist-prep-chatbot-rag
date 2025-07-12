import os
import streamlit as st
from streamlit_chat import message
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import logging
import time
import numpy as np

# Streamlit app settings - MUST be first
st.set_page_config(page_title='Disaster Preparedness RAG', layout='wide')

# Load environment variables
load_dotenv()

QDRANT_COLLECTION = "disaster_prep"
RETRIEVER_K = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_app():
    """Initialize the app components with caching"""
    return DisasterRAGApp()

class DisasterRAGApp:
    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.groq_client = None
        self.setup_clients()
    
    def setup_clients(self):
        """Initialize all required clients"""
        try:
            # Initialize embedding model
            with st.spinner("Loading embedding model..."):
                self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Initialize Qdrant client
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            qdrant_api_key = os.getenv("QDRANT_API_KEY")
            
            if qdrant_api_key:
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
            else:
                self.qdrant_client = QdrantClient(url=qdrant_url)
            
            # Initialize Langchain ChatGroq client
            groq_api_key = os.getenv("GROQ_API_KEY")
            logger.info(f"Groq API key found: {bool(groq_api_key)}")
            if groq_api_key and groq_api_key != "your_groq_api_key_here":
                self.llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
                st.success("✅ Connected to Groq API successfully!")
            else:
                st.error("❌ Groq API key not found or invalid. Please check your .env file.")
                logger.error(f"Groq API key issue: {groq_api_key}")
                
        except Exception as e:
            st.error(f"Error initializing clients: {e}")
            logger.error(f"Error initializing clients: {e}")
    
    def retrieve_context(self, query, k=RETRIEVER_K):
        """Retrieve relevant context from Qdrant"""
        try:
            # Create embedding for the query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=query_embedding.tolist(),
                limit=k,
                with_payload=True
            )
            
            # Format results
            contexts = []
            for result in search_results:
                payload = result.payload
                contexts.append({
                    'content': payload.get('content', ''),
                    'source': payload.get('source', ''),
                    'title': payload.get('title', ''),
                    'url': payload.get('url', ''),
                    'score': result.score
                })
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            st.error(f"Error retrieving context: {e}")
            return []
    
    def generate_response(self, query, context):
        """Generate response using Groq LLM"""
        if not self.llm:
            return "Groq API key not configured. Please set GROQ_API_KEY in your .env file."
        
        # Format context
        context_text = "\n\n".join([
            f"Source: {ctx['source']}\nTitle: {ctx['title']}\nContent: {ctx['content'][:500]}..."
            for ctx in context
        ])
        
        # Create prompt
        prompt = f"""You are a Disaster Preparedness & Response Assistant. Use ONLY context from FEMA, CDC, NOAA, Red Cross, WHO, UNDRR. If information is not found in the provided context, say "I don't have that information from trusted sources."

Context:
{context_text}

Question: {query}

Answer:"""
        
        try:
            # Generate response using Langchain ChatGroq
            response = self.llm.invoke(prompt)
            
            # Extract content from the response message
            return response.content
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"

# Initialize the app with caching
app = initialize_app()

# Streamlit app interface
st.title('🚨 Disaster Preparedness & Response Assistant')

st.markdown('''
📋 **This assistant provides information from trusted disaster preparedness sources:**
- FEMA (Federal Emergency Management Agency)
- Ready.gov (National Preparedness Information)
- CDC (Centers for Disease Control and Prevention)
- NOAA (National Oceanic and Atmospheric Administration)
- Red Cross (American Red Cross)
- WHO (World Health Organization)
- UNDRR (United Nations Office for Disaster Risk Reduction)

💡 **Ask questions about:**
- Emergency preparedness plans
- Disaster response procedures
- Safety guidelines
- Recovery strategies
''')

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ Information")
    st.markdown("""
    **How it works:**
    1. Enter your disaster preparedness question
    2. The system searches trusted sources
    3. AI provides answer based on retrieved context
    
    **Example Questions:**
    - How do I prepare for a hurricane?
    - What should be in an emergency kit?
    - How to stay safe during an earthquake?
    - What are evacuation procedures?
    """)

# Main query interface
query = st.text_input(
    'Enter your disaster preparedness question:', 
    placeholder="e.g., How do I prepare for a hurricane?"
)

if query:
    with st.spinner("Searching for relevant information..."):
        # Retrieve context from Qdrant
        logger.info(f"Retrieving context for query: {query}")
        contexts = app.retrieve_context(query)
    
    if contexts:
        # Show retrieved context in expander
        with st.expander(f"📚 Retrieved Context ({len(contexts)} sources)", expanded=False):
            for i, ctx in enumerate(contexts, 1):
                st.markdown(f"**Source {i}: {ctx['source']} - {ctx['title'][:60]}...**")
                st.markdown(f"*Relevance Score: {ctx['score']:.3f}*")
                st.text(ctx['content'][:300] + "..." if len(ctx['content']) > 300 else ctx['content'])
                st.markdown(f"🔗 [Original Source]({ctx['url']})")
                st.divider()
        
        # Generate and display response
        with st.spinner("Generating response..."):
            response = app.generate_response(query, contexts)
        
        st.subheader("🤖 AI-Assisted Answer")
        st.markdown(response)
        
        # Show sources used
        st.subheader("📖 Sources Referenced")
        sources = set([ctx['source'] for ctx in contexts])
        source_cols = st.columns(len(sources) if len(sources) <= 4 else 4)
        for i, source in enumerate(sources):
            with source_cols[i % 4]:
                st.info(f"🏛️ {source}")
    
    else:
        st.warning("😕 No relevant information found in the disaster preparedness database. Please try rephrasing your question or asking about general emergency preparedness topics.")

else:
    st.info("👆 Enter a question above to get started!")
    
    # Show example queries
    st.subheader("🔍 Example Questions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("How do I prepare for a hurricane?"):
            st.rerun()
        if st.button("What should be in an emergency kit?"):
            st.rerun()
    
    with col2:
        if st.button("How to stay safe during an earthquake?"):
            st.rerun()
        if st.button("What are flood safety measures?"):
            st.rerun()


