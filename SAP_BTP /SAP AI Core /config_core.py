# Configuration settings for ticket duplicate detection system
# System Settings
BASE_DIR = None  # Will be set programmatically
DATASET_DIR = None  # Will be set programmatically

# AI Model Settings
AI_MODEL = "gpt-4o"

# Vector Database Settings
VECTOR_DB_ENABLED = True
EMBEDDING_MODEL = "text-embedding-3"  # SAP BTP embedding model
VECTOR_DB_TYPE = "faiss"  # Options: "faiss"
INDEX_SAVE_LOCATION = "./index_storage"  # Location to save local Faiss index
INDEX_TYPE = "IndexFlatL2"  # Specify Faiss index type - common options include "IndexFlatL2", "IndexIVFFlat", etc.

# Duplicate Detection Settings
# Thresholds for determining duplicate status
DUPLICATE_THRESHOLDS = {
    "definite_duplicate": 0.9,  # Score >= 0.9: Definite duplicate
    "likely_duplicate": 0.8,    # Score >= 0.8 and < 0.9: Likely duplicate
    "not_duplicate": 0.8        # Score < 0.8: Not a duplicate
}

# Feature Flags
ENABLE_LLM_DUPLICATE_DETECTION = True
ENABLE_VECTOR_DB_DUPLICATE_DETECTION = True
ENABLE_PRE_FILTERING = True  # Enable pre-filtering by company code and component
ENABLE_GLOBAL_OUTAGE_DETECTION = True

# Pre-filtering Settings
PRE_VALIDATION_FIELDS = ["company_code", "component"]  # Fields to validate before processing

# Search Settings
MAX_HISTORICAL_TICKETS = 20
RECENT_TICKETS_DAYS = 30

# Fallback Mechanism
ENABLE_FALLBACK = True  # Enable fallback to manual verification if primary method fails
MAX_LLM_RETRIES = 3  # Number of retries if LLM fails

# LLM API Settings
AICORE_CLIENT_ID = ""
AICORE_CLIENT_SECRET = ""
AICORE_AUTH_URL = ""
AICORE_BASE_URL = ""
AICORE_RESOURCE_GROUP = ""
