# 🚨 Disaster Preparedness RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system that provides reliable disaster preparedness information from trusted sources including FEMA, Ready.gov, CDC, NOAA, Red Cross, WHO, and UNDRR.

## 🌟 Features

- **Trusted Data Sources**: Scrapes information from 7 authoritative disaster preparedness organizations
- **Smart Chunking**: Splits documents into ~500-token chunks with overlap for optimal retrieval
- **Semantic Search**: Uses HuggingFace sentence transformers for embedding creation
- **Vector Storage**: Stores embeddings in Qdrant with metadata (source, date, type)
- **AI-Powered Responses**: Uses Groq's LLaMA/Mistral models for response generation
- **Interactive UI**: Streamlit web application with source attribution

## 🏗️ Architecture

```
Web Scraping → Document Processing → Embedding Creation → Vector Storage → RAG Application
     ↓                ↓                    ↓              ↓              ↓
  Selenium      Text Chunking        SentenceTransformer  Qdrant     Streamlit + Groq
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- Chrome browser (for web scraping)
- Qdrant instance (local or cloud)
- Groq API key

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd disaster-prep-rag
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Copy the environment template and fill in your API keys:

```bash
copy .env.example .env
```

Edit `.env` with your actual values:

```env
# Groq API Key (get from: https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here  # Optional for local instances

# Optional: HuggingFace API Token
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

### 4. Start Qdrant (Local Installation)

#### Option A: Docker
```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Option B: Local Installation
```bash
# Download and run Qdrant locally
curl -L https://github.com/qdrant/qdrant/releases/download/v1.7.0/qdrant-x86_64-pc-windows-gnu.zip -o qdrant.zip
unzip qdrant.zip
./qdrant.exe
```

## 🚀 Usage

### Step 1: Scrape Data

Collect disaster preparedness information from trusted sources:

```bash
python scraper.py
```

This will:
- Scrape content from FEMA, Ready.gov, CDC, NOAA, Red Cross, WHO, UNDRR
- Save raw data to `data/scraped_disaster_prep_data.json`
- Take approximately 10-15 minutes to complete

### Step 2: Process and Create Embeddings

Convert documents to embeddings and upload to Qdrant:

```bash
python process_embeddings.py
```

This will:
- Split documents into ~500-character chunks with 50-character overlap
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Upload to Qdrant collection named 'disaster_prep'
- Process may take 20-30 minutes depending on data volume

### Step 3: Launch RAG Application

Start the Streamlit web interface:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📋 Usage Examples

### Example Questions You Can Ask:

- "How do I prepare for a hurricane?"
- "What should be in an emergency kit?"
- "How to stay safe during an earthquake?"
- "What are evacuation procedures for floods?"
- "How to create a family emergency plan?"
- "What supplies do I need for 72 hours?"

### Expected Response Format:

The AI assistant will:
1. Search the vector database for relevant information
2. Retrieve top 5 most relevant chunks
3. Generate a response using only the retrieved context
4. Provide source attribution
5. Say "I don't have that information from trusted sources" if no relevant context is found

## 🔧 Configuration

### Customizing Chunk Size

Edit `process_embeddings.py`:

```python
class DocumentProcessor:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.chunk_size = 500  # Adjust chunk size
        self.overlap = 50      # Adjust overlap
```

### Changing Embedding Model

Edit both `process_embeddings.py` and `app.py`:

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Alternative model
```

### Adjusting Retrieval Parameters

Edit `app.py`:

```python
RETRIEVER_K = 10  # Retrieve more contexts
```

### Switching LLM Models

Edit `app.py` in the `generate_response` method:

```python
model="mixtral-8x7b-32768",  # Switch to Mixtral
# or
model="llama3-70b-8192",     # Switch to larger LLaMA
```

## 📁 Project Structure

```
disaster-prep-rag/
├── scraper.py                 # Web scraping module
├── process_embeddings.py      # Document processing and embedding creation
├── app.py                     # Streamlit RAG application
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore patterns
├── README.md                 # This file
└── data/                     # Generated data directory
    └── scraped_disaster_prep_data.json
```

## 🎯 Data Sources

| Source | Organization | Content Type |
|--------|-------------|--------------|
| FEMA | Federal Emergency Management Agency | Emergency management, planning, response |
| Ready.gov | National Preparedness Information | Personal preparedness, emergency kits |
| CDC | Centers for Disease Control | Health-related disaster preparedness |
| NOAA | National Oceanic and Atmospheric Administration | Weather-related safety |
| Red Cross | American Red Cross | Emergency response, first aid |
| WHO | World Health Organization | Global health emergencies |
| UNDRR | UN Office for Disaster Risk Reduction | International disaster risk reduction |

## 🛠️ Troubleshooting

### Common Issues

**1. Chrome Driver Issues**
```bash
# The scraper will automatically download ChromeDriver
# If issues persist, manually download and set path in .env
CHROME_DRIVER_PATH=/path/to/chromedriver
```

**2. Qdrant Connection Errors**
```bash
# Ensure Qdrant is running
docker ps  # Check if container is running
# Or restart Qdrant service
```

**3. Groq API Rate Limits**
- The free tier has rate limits
- Add delays between requests if needed
- Consider upgrading to paid tier for production use

**4. Memory Issues During Embedding Creation**
- Process smaller batches of documents
- Use a smaller embedding model
- Increase system RAM

### Debug Mode

Enable detailed logging by setting:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 🔄 Updates and Maintenance

### Refreshing Data

To update the disaster preparedness information:

1. Delete existing data: `rm -rf data/`
2. Re-run scraper: `python scraper.py`
3. Re-process embeddings: `python process_embeddings.py`

### Adding New Sources

1. Edit `scraper.py` and add new source configuration
2. Update the source validation in `app.py`
3. Re-run the complete pipeline

## 📊 Performance Metrics

- **Scraping Speed**: ~2-3 seconds per page
- **Embedding Creation**: ~100 chunks per minute
- **Query Response Time**: 1-3 seconds
- **Storage**: ~100MB per 1000 documents
- **Accuracy**: Based on quality of source material and retrieval relevance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


