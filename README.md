# 📚 LlamaIndex Document Search with BM25 & HyDE

A powerful document search application that combines BM25 retrieval with HyDE (Hypothetical Document Embeddings) and GPT-4o for intelligent document querying.

## 🚀 Features

- **BM25 Retrieval**: Fast and efficient keyword-based document search
- **HyDE Enhancement**: Query expansion using hypothetical document embeddings
- **GPT-4o Integration**: Advanced answer generation from retrieved context
- **Hierarchical Chunking**: Multi-level document segmentation (4096, 2048, 1024, 512 tokens)
- **Multiple File Formats**: Supports `.txt`, `.pdf`, `.docx` files
- **Streamlit Interface**: User-friendly web interface

## 🛠️ Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd llamaIndexReRankRag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Copy your API keys to `.env` file or Streamlit secrets
   - Required: `OPENAI_API_KEY`

4. Add your documents:
   - Place your documents in the `./my_local_docs/` folder
   - Supports `.txt`, `.pdf`, and `.docx` files

### Local Development

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Add your secrets in the Streamlit Cloud dashboard:
   - Go to your app settings
   - Add all the secrets from `.streamlit/secrets.toml`
   - Deploy!

### Required Secrets for Deployment

```toml
OPENAI_API_KEY = "your-openai-api-key"
# Add other secrets as needed
```

## 📖 How It Works

1. **Document Loading**: Reads all files from `./my_local_docs/`
2. **Hierarchical Chunking**: Splits documents into multiple chunk sizes
3. **BM25 Indexing**: Creates a BM25 index for fast keyword search
4. **HyDE Query Enhancement**: Expands queries with hypothetical documents
5. **GPT-4o Answer Generation**: Synthesizes answers from retrieved chunks

## 🔧 Configuration

- **Similarity Top K**: Number of chunks to retrieve (default: 20)
- **Chunk Sizes**: Hierarchical chunking levels (4096, 2048, 1024, 512)
- **Model**: GPT-4o for answer generation
- **Temperature**: 0 for consistent responses

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── secrets.toml      # Streamlit secrets template
├── my_local_docs/        # Your documents directory
├── .env                  # Environment variables (local)
├── .gitignore           # Git ignore file
└── README.md            # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Support

If you encounter any issues:
1. Check that all dependencies are installed
2. Verify your API keys are correctly set
3. Ensure documents are in the `./my_local_docs/` folder
4. Check the Streamlit logs for error messages 