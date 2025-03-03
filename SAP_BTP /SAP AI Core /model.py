import os
import json
import numpy as np
import faiss
import re
from datetime import datetime, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Make sure to download NLTK resources when setting up
# nltk.download('punkt')
# nltk.download('stopwords')

# Ticket quality checker constants
MIN_SUMMARY_LENGTH = 10
MIN_DESCRIPTION_LENGTH = 30
MIN_UNIQUE_WORDS = 15
REQUIRED_FIELDS = [
    r'(?i)steps|step[ -]by[ -]step|procedure|process',  # Steps to reproduce
    r'(?i)expect|expected|should|supposed to',          # Expected behavior
    r'(?i)actual|instead|but|however'                   # Actual behavior
]

# Load stopwords
STOP_WORDS = set(stopwords.words('english'))

def check_summary(summary):
    """Check if the summary is sufficient."""
    if not summary:
        return False, "Summary is missing."
        
    if len(summary) < MIN_SUMMARY_LENGTH:
        return False, f"Summary is too short. It should be at least {MIN_SUMMARY_LENGTH} characters."
    
    # Check if summary is just generic text
    generic_phrases = ['issue', 'problem', 'error', 'bug', 'help needed']
    if summary.lower() in generic_phrases:
        return False, "Summary is too generic. Please be more specific."
        
    return True, "Summary is sufficient."

def check_description(description):
    """Check if the description is sufficient."""
    if not description:
        return False, "Description is missing."
        
    if len(description) < MIN_DESCRIPTION_LENGTH:
        return False, f"Description is too short. It should be at least {MIN_DESCRIPTION_LENGTH} characters."
    
    # Check for unique words
    tokens = word_tokenize(description.lower())
    meaningful_words = [word for word in tokens if word.isalnum() and word not in STOP_WORDS]
    unique_words = set(meaningful_words)
    
    if len(unique_words) < MIN_UNIQUE_WORDS:
        return False, f"Description lacks detail. Please use at least {MIN_UNIQUE_WORDS} unique meaningful words."
    
    # Check for required information patterns
    missing_patterns = []
    for pattern in REQUIRED_FIELDS:
        if not re.search(pattern, description):
            if 'steps' in pattern:
                missing_patterns.append("steps to reproduce")
            elif 'expect' in pattern:
                missing_patterns.append("expected behavior")
            elif 'actual' in pattern:
                missing_patterns.append("actual behavior")
    
    if missing_patterns:
        return False, f"Description is missing: {', '.join(missing_patterns)}."
        
    return True, "Description is sufficient."

def is_ticket_sufficient(summary, description):
    """
    Check if the ticket has sufficient information.
    
    Parameters:
    -----------
    summary : str
        The ticket summary/title
    description : str
        The ticket description/body
        
    Returns:
    --------
    tuple
        (is_sufficient, message, details)
        - is_sufficient: boolean indicating if the ticket is sufficient
        - message: overall result message
        - details: dictionary with detailed results for each check
    """
    summary_result, summary_message = check_summary(summary)
    description_result, description_message = check_description(description)
    
    details = {
        "summary": {
            "is_sufficient": summary_result,
            "message": summary_message
        },
        "description": {
            "is_sufficient": description_result,
            "message": description_message
        }
    }
    
    is_sufficient = summary_result and description_result
    
    if is_sufficient:
        message = "Ticket has sufficient information and can be processed."
    else:
        message = "Ticket information is insufficient. Please provide more details."
        
    return is_sufficient, message, details

# Initialize AI model
def init_models():
    """Initialize AI models for embeddings and LLM."""
    proxy_client = get_proxy_client("gen-ai-hub")
    llm = init_llm(config.AI_MODEL, proxy_client=proxy_client)
    embeddings = init_embeddings(config.EMBEDDING_MODEL, proxy_client=proxy_client)
    return llm, embeddings

# Define prompt templates
DUPLICATE_CHECK_TEMPLATE = """You are a ticket analysis system. Your task is to determine if a new ticket is a duplicate of any existing tickets.

Historical Tickets:
{historical_tickets}

New Ticket Details:
Company Code: {company_code}
Component: {component}
Summary: {summary}
Description: {description}

Instructions:
1. Focus ONLY on semantic similarity in the problem description
2. Company code and component matching has already been validated and filtered
3. A ticket is likely duplicate if it describes the same issue with similar symptoms or errors
4. Do NOT perform additional validations - focus solely on content similarity analysis

Respond in this EXACT format (replace values appropriately):
{{"is_duplicate": true/false, "original_ticket_id": "TICKET-123 or null", "confidence": 0.95, "reasoning": "Explanation here"}}
"""

def get_embedding(text, embedding_model):
    """Get embedding vector for text using SAP BTP text-embedding-3 model."""
    # Normalize the text by removing excessive whitespace and converting to lowercase
    normalized_text = " ".join(text.lower().split())

    # Call the SAP BTP API to get the embedding
    embedding = embedding_model.embed_documents([normalized_text])[0]

    # Convert the embedding to a numpy array
    return np.array(embedding).astype(np.float32)

def save_faiss_index(index, filename="faiss_index"):
    """Save FAISS index to disk."""
    save_path = os.path.join(config.INDEX_SAVE_LOCATION, filename)

    # Create directory if it doesn't exist
    os.makedirs(config.INDEX_SAVE_LOCATION, exist_ok=True)

    faiss.write_index(index, save_path)
    return save_path

def load_faiss_index(filename="faiss_index"):
    """Load FAISS index from disk."""
    load_path = os.path.join(config.INDEX_SAVE_LOCATION, filename)

    if os.path.exists(load_path):
        return faiss.read_index(load_path)
    return None

def create_vector_index(data_records, embeddings_model):
    """Create FAISS index from ticket data."""
    # First check if we have a saved index
    index = load_faiss_index()
    if index is not None:
        # If index exists, we could update it with new records
        # For simplicity, we'll recreate it in this example
        pass

    # Extract text data for embedding
    texts = []
    for record in data_records:
        # Combine summary and description for embedding
        text = f"{record['summary']} {record['description']}"
        texts.append(text)

    # Generate embeddings
    embeddings = np.array([get_embedding(text, embeddings_model) for text in texts]).astype(np.float32)

    # Create FAISS index based on config
    embedding_dim = embeddings.shape[1]

    # Set index type based on config, default to flat index
    if config.INDEX_TYPE == "IndexFlatL2":
        index = faiss.IndexFlatL2(embedding_dim)
    else:
        # Default to HNSW index
        index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 connections per node
        index.hnsw.efConstruction = 40  # Higher values give better recall but slower construction
        index.hnsw.efSearch = 16  # Higher values give better recall but slower search

    # Add vectors to index
    if len(embeddings) > 0:
        index.add(embeddings)

    # Save the index
    save_faiss_index(index)

    return index, embeddings

def search_vector_db(index, query_embedding, k=5):
    """Search vector database for similar tickets."""
    # Reshape query embedding to match FAISS expectations
    query_embedding = query_embedding.reshape(1, -1)

    # Limit k to the number of vectors in the index to avoid out-of-bounds errors
    k = min(k, index.ntotal)

    # If index is empty, return empty results
    if k == 0:
        return np.array([]), np.array([])

    # Search the index
    distances, indices = index.search(query_embedding, k)

    return distances[0], indices[0]  # Return first result's distances and indices

def format_tickets_for_context(df, max_tickets=None):
    """Format recent tickets as context for the LLM."""
    if max_tickets is None:
        max_tickets = config.MAX_HISTORICAL_TICKETS

    # Sort by date if available
    if 'closure_date' in df.columns:
        df = df.sort_values('closure_date', ascending=False)

    recent_tickets = df.head(max_tickets)
    formatted_tickets = []

    for _, row in recent_tickets.iterrows():
        ticket = (
            f"Ticket ID: {row['ticket_id']}\n"
            f"Company: {row['company_code']}\n"
            f"Component: {row['component']}\n"
            f"Summary: {row['summary']}\n"
            f"Description: {row['description']}\n"
            "---"
        )
        formatted_tickets.append(ticket)

    return "\n\n".join(formatted_tickets)

def parse_llm_response(response_text, default_response):
    """Safely parse LLM response and ensure it matches expected format."""
    try:
        # Clean up common JSON formatting issues
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "")

        # Parse JSON
        parsed = json.loads(response_text)

        # Validate required fields
        if "is_duplicate" not in parsed and "is_global_outage" not in parsed:
            return default_response

        return parsed
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        print(f"Raw response: {response_text}")
        return default_response

def validate_ticket_data(ticket_data):
    """Pre-validate ticket data before processing."""
    # Check for required fields
    for field in config.PRE_VALIDATION_FIELDS:
        if not ticket_data.get(field):
            return False, f"Missing required field: {field}"

    return True, "Validation successful"

def check_duplicate(master_df, ticket_data):
    """Check if a ticket is duplicate using direct LLM analysis or Vector DB based on config."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }

    # Initialize models
    llm, embeddings = init_models()

    # Pre-validation: Only process tickets with valid company code and component
    if config.ENABLE_PRE_FILTERING:
        is_valid, validation_message = validate_ticket_data(ticket_data)
        if not is_valid:
            return {
                "is_duplicate": False,
                "original_ticket_id": None,
                "confidence": 0.0,
                "reasoning": validation_message
            }
    
    # NEW: Check if ticket content is sufficient before proceeding to duplicate detection
    is_sufficient, message, details = is_ticket_sufficient(
        str(ticket_data.get('summary', '')), 
        str(ticket_data.get('description', ''))
    )
    
    if not is_sufficient:
        return {
            "is_duplicate": False,
            "original_ticket_id": None,
            "confidence": 0.0,
            "reasoning": f"Insufficient ticket information: {message}",
            "details": details  # Provide detailed feedback
        }

    try:
        # First try Vector DB approach if enabled
        if config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION:
            # Pre-filter based on company code and component before searching
            relevant_df = master_df[
                (master_df['company_code'] == ticket_data['company_code']) &
                (master_df['component'] == ticket_data['component'])
            ].copy()

            # Skip to LLM if no relevant tickets found in Vector DB
            if not relevant_df.empty:
                # Create vector index for relevant tickets
                relevant_records = relevant_df.to_dict(orient="records")
                vector_index, vector_embeddings = create_vector_index(relevant_records, embeddings)

                # Generate embedding for new ticket
                query_text = f"{ticket_data['summary']} {ticket_data['description']}"
                query_embedding = get_embedding(query_text, embeddings)

                # Search for similar tickets
                distances, indices = search_vector_db(vector_index, query_embedding, k=min(5, len(relevant_records)))

                # Skip if no results returned
                if indices.size == 0:
                    return default_response

                # FIXED: For L2 distances, lower values indicate higher similarity
                # FIXED: Avoid division by zero by adding small epsilon
                epsilon = 1e-10
                # Use inverse of distance as similarity (higher similarity for smaller distance)
                max_distance = np.max(distances) if distances.size > 0 else 1.0
                # Avoid division by zero
                max_distance = max(max_distance, epsilon)

                # Calculate similarities - properly account for L2 distance metric
                similarities = []
                for dist in distances:
                    # Normalize distance to [0,1] range and invert
                    if dist == 0:  # Perfect match
                        similarities.append(1.0)
                    else:
                        norm_dist = dist / max_distance
                        similarities.append(1.0 - norm_dist)

                # Find the best match (highest similarity)
                best_idx = -1
                best_similarity = 0

                for i, similarity in enumerate(similarities):
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = i

                # FIXED: Get the matching ticket details - use indices to lookup in relevant_df, not master_df
                if best_idx >= 0 and best_idx < len(indices) and indices[best_idx] < len(relevant_df):
                    # Get the actual index in the filtered dataframe
                    relevant_idx = indices[best_idx]

                    # Safely access the dataframe
                    if 0 <= relevant_idx < len(relevant_df):
                        matching_ticket = relevant_df.iloc[relevant_idx]

                        # Determine duplicate status based on thresholds
                        if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate']:
                            return {
                                "is_duplicate": True,
                                "original_ticket_id": matching_ticket['ticket_id'],
                                "confidence": best_similarity,
                                "reasoning": f"Vector similarity score: {best_similarity:.4f} - Definite duplicate"
                            }
                        elif best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate']:
                            return {
                                "is_duplicate": True,
                                "original_ticket_id": matching_ticket['ticket_id'],
                                "confidence": best_similarity,
                                "reasoning": f"Vector similarity score: {best_similarity:.4f} - Likely duplicate"
                            }

        # Fall back to LLM approach if Vector DB not enabled or didn't find a match
        if config.ENABLE_LLM_DUPLICATE_DETECTION:
            # Pre-filter for LLM to reduce complexity
            relevant_df = master_df[
                (master_df['company_code'] == ticket_data['company_code']) &
                (master_df['component'] == ticket_data['component'])
            ].copy()

            # Only use LLM if we have relevant tickets
            if relevant_df.empty:
                return {
                    "is_duplicate": False,
                    "original_ticket_id": None,
                    "confidence": 0.0,
                    "reasoning": "No matching company code and component found"
                }

            historical_context = format_tickets_for_context(relevant_df)

            # Prepare and execute the chain
            prompt = ChatPromptTemplate.from_template(DUPLICATE_CHECK_TEMPLATE)
            chain = prompt | llm | StrOutputParser()

            # Implement retry logic
            retries = 0
            while retries < config.MAX_LLM_RETRIES:
                try:
                    response = chain.invoke({
                        "historical_tickets": historical_context,
                        "company_code": str(ticket_data["company_code"]),
                        "component": str(ticket_data["component"]),
                        "summary": str(ticket_data["summary"]),
                        "description": str(ticket_data["description"])
                    })

                    result = parse_llm_response(response, default_response)
                    if result != default_response:
                        return result

                    retries += 1
                except Exception as e:
                    print(f"LLM retry {retries+1}/{config.MAX_LLM_RETRIES} failed: {str(e)}")
                    retries += 1

            # If we get here, all retries failed
            return default_response

        # If neither Vector DB nor LLM analysis is enabled
        return default_response

    except Exception as e:
        print(f"Error in check_duplicate: {str(e)}")
        return default_response
