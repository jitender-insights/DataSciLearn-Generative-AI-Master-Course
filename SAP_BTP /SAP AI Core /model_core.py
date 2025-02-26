import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import faiss
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.init_models import init_llm
import config

# Environment setup
os.environ['AICORE_CLIENT_ID'] = config.AICORE_CLIENT_ID
os.environ['AICORE_CLIENT_SECRET'] = config.AICORE_CLIENT_SECRET
os.environ['AICORE_AUTH_URL'] = config.AICORE_AUTH_URL
os.environ['AICORE_BASE_URL'] = config.AICORE_BASE_URL
os.environ['AICORE_RESOURCE_GROUP'] = config.AICORE_RESOURCE_GROUP
load_dotenv()

# Define dataset directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Update config with paths
config.BASE_DIR = BASE_DIR
config.DATASET_DIR = DATASET_DIR

# Initialize AI model
proxy_client = get_proxy_client("gen-ai-hub")
llm = init_llm(config.AI_MODEL, proxy_client=proxy_client)

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

GLOBAL_OUTAGE_TEMPLATE = """You are a ticket analysis system. Determine if this ticket indicates a global outage.

Historical Global Outages:
{global_outages}

New Ticket:
Summary: {summary}
Description: {description}

Respond in this EXACT format (replace values appropriately):
{{"is_global_outage": true/false, "confidence": 0.95, "reasoning": "Explanation here"}}"""

def get_embedding(text, embedding_model=config.EMBEDDING_MODEL):
    """Get embedding vector for text using SAP BTP text-embedding-3 model."""
    # In a real implementation, this would call the SAP BTP embedding API
    # For this example, we'll use a placeholder function that returns random embeddings
    
    # Normalize the text by removing excessive whitespace and converting to lowercase
    normalized_text = " ".join(text.lower().split())
    
    # This is a placeholder - in production, call the actual embedding API
    # The dimension should match what SAP BTP text-embedding-3 provides
    embedding_dim = 1536  # Typical dimension for modern embedding models
    return np.random.rand(embedding_dim).astype(np.float32)

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

def create_vector_index(data_records):
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
    embeddings = np.array([get_embedding(text) for text in texts]).astype(np.float32)
    
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
    
    # Pre-validation: Only process tickets with valid company code and component
    is_valid, validation_message = validate_ticket_data(ticket_data)
    if not is_valid:
        return {
            "is_duplicate": False,
            "original_ticket_id": None,
            "confidence": 0.0,
            "reasoning": validation_message
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
                vector_index, embeddings = create_vector_index(relevant_records)
                
                # Generate embedding for new ticket
                query_text = f"{ticket_data['summary']} {ticket_data['description']}"
                query_embedding = get_embedding(query_text)
                
                # Search for similar tickets
                distances, indices = search_vector_db(vector_index, query_embedding, k=5)
                
                # Convert distances to cosine similarity scores
                # For L2 distances, convert to similarity (1 - normalized_distance)
                max_distance = np.max(distances) if distances.size > 0 else 1.0
                similarities = [1 - (dist/max_distance) for dist in distances]
                
                # Find the best match
                best_idx = 0  # Default to first result
                best_similarity = similarities[0] if similarities else 0
                
                for i, similarity in enumerate(similarities):
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = i
                
                # Get the matching ticket details if there are results
                if indices.size > 0 and best_idx < len(indices):
                    matching_ticket = relevant_df.iloc[indices[best_idx]]
                    
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

def check_recently_closed_tickets(ticket_data, master_df):
    """Check if a similar ticket was closed recently (within configured days)."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "No recent similar tickets found"
    }
    
    # Pre-validation
    is_valid, validation_message = validate_ticket_data(ticket_data)
    if not is_valid:
        return default_response
    
    try:
        # Pre-filter by company code and component first
        cutoff_date = datetime.now() - timedelta(days=config.RECENT_TICKETS_DAYS)
        recent_df = master_df[
            (master_df["closure_date"] >= cutoff_date) &
            (master_df['company_code'] == ticket_data['company_code']) &
            (master_df['component'] == ticket_data['component'])
        ].copy()
        
        if recent_df.empty:
            return default_response
        
        # Decide which approach to use based on configuration
        if config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION:
            # Create vector index for recent tickets
            recent_records = recent_df.to_dict(orient="records")
            vector_index, embeddings = create_vector_index(recent_records)
            
            # Generate embedding for new ticket
            query_text = f"{ticket_data['summary']} {ticket_data['description']}"
            query_embedding = get_embedding(query_text)
            
            # Search in vector database
            distances, indices = search_vector_db(vector_index, query_embedding, k=5)
            
            # Convert distances to similarity scores
            max_distance = np.max(distances) if distances.size > 0 else 1.0
            similarities = [1 - (dist/max_distance) for dist in distances]
            
            # Find the best match
            if not similarities:
                return default_response
                
            best_idx = 0
            best_similarity = similarities[0]
            
            for i, similarity in enumerate(similarities):
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_idx = i
            
            # Determine duplicate status based on thresholds
            if indices.size > 0 and best_idx < len(indices):
                if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate'] or best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate']:
                    # Get the matching ticket details
                    matching_ticket = recent_df.iloc[indices[best_idx]]
                    
                    return {
                        "is_duplicate": True,
                        "original_ticket_id": matching_ticket['ticket_id'],
                        "confidence": best_similarity,
                        "reasoning": f"Recently closed ticket with vector similarity: {best_similarity:.4f}"
                    }
            
            return default_response
        
        elif config.ENABLE_LLM_DUPLICATE_DETECTION:
            # Use LLM-based approach as fallback
            historical_context = format_tickets_for_context(recent_df)
            
            prompt = ChatPromptTemplate.from_template(DUPLICATE_CHECK_TEMPLATE)
            chain = prompt | llm | StrOutputParser()
            
            # Implement retry logic for LLM
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
            
            return default_response
        
        else:
            return default_response
            
    except Exception as e:
        print(f"Error in check_recently_closed_tickets: {str(e)}")
        return default_response

def is_global_outage(ticket_data, master_df):
    """Check if ticket matches an ongoing global outage issue."""
    if not config.ENABLE_GLOBAL_OUTAGE_DETECTION:
        return False
        
    try:
        global_outages = master_df[master_df["status"] == "global_outage"].copy()
        
        if global_outages.empty:
            return False
        
        global_outages_context = format_tickets_for_context(global_outages)
        
        prompt = ChatPromptTemplate.from_template(GLOBAL_OUTAGE_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        
        # Implement retry logic
        retries = 0
        while retries < config.MAX_LLM_RETRIES:
            try:
                response = chain.invoke({
                    "global_outages": global_outages_context,
                    "summary": str(ticket_data["summary"]),
                    "description": str(ticket_data["description"])
                })
                
                result = parse_llm_response(response, {"is_global_outage": False})
                return result.get("is_global_outage", False)
                
            except Exception as e:
                print(f"LLM retry {retries+1}/{config.MAX_LLM_RETRIES} failed: {str(e)}")
                retries += 1
        
        return False
        
    except Exception as e:
        print(f"Error in is_global_outage: {str(e)}")
        return False

def load_master_data():
    """Load master ticket data."""
    file_path = os.path.join(DATASET_DIR, "master_data.csv")
    df = pd.read_csv(file_path)
    df.rename(columns={
        "Incident Key": "ticket_id",
        "Summary": "summary",
        "Custom Field (Company)": "company_code",
        "Custom Field (Component)": "component",
        "Incident Type": "incident_type",
        "Description": "description",
        "Status": "status",
        "Resolved_Parsed": "closure_date"
    }, inplace=True)
    df["closure_date"] = pd.to_datetime(df["closure_date"], errors="coerce")
    return df

def load_new_ticket_data():
    """Load new ticket data."""
    file_path = os.path.join(DATASET_DIR, "new_ticket_updated.csv")
    df = pd.read_csv(file_path)
    df.rename(columns={
        "Incident Key": "ticket_id",
        "Summary": "summary",
        "Custom Field (Company)": "company_code",
        "Custom Field (Component)": "component",
        "Incident Type": "incident_type",
        "Description": "description",
    }, inplace=True)
    return df.to_dict(orient="records")

def create_subtask(original_ticket_id, duplicate_data):
    """Create a subtask for a duplicate ticket."""
    return {
        "parent_ticket_id": original_ticket_id,
        "summary": f"Duplicate of {original_ticket_id}",
        "description": f"Original description: {duplicate_data['description']}",
        "status": "Created from duplicate",
        "created_date": datetime.now().isoformat()
    }
