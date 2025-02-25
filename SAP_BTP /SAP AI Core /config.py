# config.py

# Duplicate Detection Thresholds
DUPLICATE_THRESHOLDS = {
    "definite_duplicate": 0.9,  # Tickets with similarity >= 0.9 are definite duplicates
    "likely_duplicate": 0.8,    # Tickets with similarity between 0.8 and 0.9 are likely duplicates
    "not_duplicate": 0.8        # Tickets with similarity < 0.8 are not duplicates
}

# Enable/Disable LLM and Vector DB
ENABLE_LLM = True  # Set to False to disable LLM and rely only on Vector DB
ENABLE_VECTOR_DB = True  # Set to False to disable Vector DB and rely only on LLM

# Vector DB Configuration
VECTOR_DB_CONFIG = {
    "index_type": "HNSW",  # Options: "HNSW" or "Faiss"
    "embedding_model": "text-embedding-3",  # SAP BTP embedding model
    "similarity_metric": "cosine",  # Similarity metric for Vector DB
    "pre_filter_fields": ["company_code", "component"]  # Fields for pre-filtering
}

# LLM Configuration
LLM_CONFIG = {
    "model_name": "gpt-4o",  # LLM model to use
    "max_tokens": 1000,  # Maximum tokens for LLM response
    "temperature": 0.7  # Temperature for LLM response
}

# File Paths
DATASET_DIR = "dataset"  # Directory containing dataset files
MASTER_DATA_FILE = "master_data.csv"  # Master ticket data file
NEW_TICKET_FILE = "new_ticket_updated.csv"  # New ticket data file
