import os
import pandas as pd
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.init_models import init_llm
import faiss  # For Vector DB
import numpy as np
import openai  # For OpenAI embeddings
from config import (
    DUPLICATE_THRESHOLDS, ENABLE_LLM, ENABLE_VECTOR_DB, VECTOR_DB_CONFIG, LLM_CONFIG, DATASET_DIR, MASTER_DATA_FILE, NEW_TICKET_FILE
)

# Environment setup
os.environ['AICORE_CLIENT_ID'] = " "  # Add your SAP BTP client ID
os.environ['AICORE_CLIENT_SECRET'] = " "  # Add your SAP BTP client secret
os.environ['AICORE_AUTH_URL'] = " "  # Add your SAP BTP auth URL
os.environ['AICORE_BASE_URL'] = " "  # Add your SAP BTP base URL
os.environ['AICORE_RESOURCE_GROUP'] = " "  # Add your SAP BTP resource group
load_dotenv()

# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")  # Add your OpenAI API key to .env

# Define dataset directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Initialize AI model
proxy_client = get_proxy_client("gen-ai-hub")
llm = init_llm(LLM_CONFIG["model_name"], proxy_client=proxy_client)

# Initialize Vector DB with Faiss (Inner Product for cosine similarity)
if ENABLE_VECTOR_DB:
    embedding_dim = 1536  # OpenAI text-embedding-3-small uses 1536 dimensions
    index_file = os.path.join(DATASET_DIR, "faiss_index.index")
    
    if os.path.exists(index_file):
        # Load existing index from disk
        index = faiss.read_index(index_file)
    else:
        # Create a new index
        index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity

# Define refined prompt templates
DUPLICATE_CHECK_TEMPLATE = """You are a ticket analysis system. Your task is to determine if a new ticket is a duplicate of any existing tickets.

Historical Tickets (Pre-filtered by Company Code and Component):
{historical_tickets}

New Ticket Details:
Company Code: {company_code}
Component: {component}
Summary: {summary}
Description: {description}

Instructions:
1. Focus on semantic similarity between the new ticket and historical tickets.
2. A ticket is a duplicate if it describes the same issue in a similar context.
3. Ignore differences in phrasing or wording if the core issue is the same.
4. Be strict about duplicate detection - only mark as duplicate if the issue is truly the same.

Respond in this EXACT format (replace values appropriately):
{{"is_duplicate": true/false, "original_ticket_id": "TICKET-123 or null", "confidence": 0.95, "reasoning": "Explanation here"}}"""

FALLBACK_DUPLICATE_CHECK_TEMPLATE = """You are a ticket analysis system. Your task is to determine if a new ticket is a duplicate of any existing tickets.

Historical Tickets (Pre-filtered by Company Code and Component):
{historical_tickets}

New Ticket Details:
Company Code: {company_code}
Component: {component}
Summary: {summary}
Description: {description}

Instructions:
1. The Vector DB did not find a match, so you need to perform a detailed semantic analysis.
2. Compare the new ticket against historical tickets for similar issues.
3. A ticket is a duplicate if it describes the same issue in a similar context.
4. Be strict about duplicate detection - only mark as duplicate if the issue is truly the same.

Respond in this EXACT format (replace values appropriately):
{{"is_duplicate": true/false, "original_ticket_id": "TICKET-123 or null", "confidence": 0.95, "reasoning": "Explanation here"}}"""

GLOBAL_OUTAGE_TEMPLATE = """You are a ticket analysis system. Determine if this ticket indicates a global outage.

Historical Global Outages:
{global_outages}

New Ticket:
Summary: {summary}
Description: {description}

Respond in this EXACT format (replace values appropriately):
{{"is_global_outage": true/false, "confidence": 0.95, "reasoning": "Explanation here"}}"""

def format_tickets_for_context(df, max_tickets=20):
    """Format recent tickets as context for the LLM."""
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
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "")
        parsed = json.loads(response_text)
        if "is_duplicate" not in parsed and "is_global_outage" not in parsed:
            return default_response
        return parsed
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        print(f"Raw response: {response_text}")
        return default_response

def generate_embedding(text):
    """Generate embeddings for text using OpenAI's text-embedding-3-small model."""
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response['data'][0]['embedding'], dtype='float32')
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def check_duplicate_with_vector_db(ticket_data, master_df):
    """Check if a ticket is duplicate using Vector DB."""
    if not ENABLE_VECTOR_DB:
        return {"is_duplicate": False, "original_ticket_id": None, "confidence": 0.0, "reasoning": "Vector DB disabled"}
    
    # Pre-filter based on company code and component
    pre_filtered_df = master_df[
        (master_df['company_code'] == ticket_data['company_code']) &
        (master_df['component'] == ticket_data['component'])
    ]
    
    if pre_filtered_df.empty:
        return {"is_duplicate": False, "original_ticket_id": None, "confidence": 0.0, "reasoning": "No matching company/component"}
    
    # Generate embeddings for the new ticket and pre-filtered tickets
    new_ticket_embedding = generate_embedding(ticket_data["description"])
    if new_ticket_embedding is None:
        return {"is_duplicate": False, "original_ticket_id": None, "confidence": 0.0, "reasoning": "Failed to generate embedding"}
    
    pre_filtered_embeddings = np.array([generate_embedding(desc) for desc in pre_filtered_df["description"]])
    pre_filtered_embeddings = np.array([emb for emb in pre_filtered_embeddings if emb is not None])  # Remove None values
    
    if len(pre_filtered_embeddings) == 0:
        return {"is_duplicate": False, "original_ticket_id": None, "confidence": 0.0, "reasoning": "No valid embeddings for historical tickets"}
    
    # Add embeddings to Vector DB index
    index.add(pre_filtered_embeddings)
    
    # Search for similar tickets in Vector DB using cosine similarity
    distances, indices = index.search(np.array([new_ticket_embedding]), k=5)
    best_match_index = indices[0][0]
    best_match_distance = distances[0][0]
    
    # Calculate confidence score (cosine similarity is normalized to [0, 1])
    confidence = (1 + best_match_distance) / 2  # Convert inner product to cosine similarity
    
    if confidence >= DUPLICATE_THRESHOLDS["definite_duplicate"]:
        return {
            "is_duplicate": True,
            "original_ticket_id": pre_filtered_df.iloc[best_match_index]["ticket_id"],
            "confidence": confidence,
            "reasoning": "High confidence match found in Vector DB"
        }
    elif confidence >= DUPLICATE_THRESHOLDS["likely_duplicate"]:
        return {
            "is_duplicate": True,
            "original_ticket_id": pre_filtered_df.iloc[best_match_index]["ticket_id"],
            "confidence": confidence,
            "reasoning": "Likely duplicate found in Vector DB"
        }
    else:
        return {"is_duplicate": False, "original_ticket_id": None, "confidence": confidence, "reasoning": "No duplicate found in Vector DB"}

def check_duplicate_with_llm(master_df, ticket_data, is_fallback=False):
    """Check if a ticket is duplicate using LLM."""
    if not ENABLE_LLM:
        return {"is_duplicate": False, "original_ticket_id": None, "confidence": 0.0, "reasoning": "LLM disabled"}
    
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }
    
    try:
        # Use fallback prompt if Vector DB did not find a match
        prompt_template = FALLBACK_DUPLICATE_CHECK_TEMPLATE if is_fallback else DUPLICATE_CHECK_TEMPLATE
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        relevant_df = master_df[
            (master_df['company_code'] == ticket_data['company_code']) |
            (master_df['component'] == ticket_data['component'])
        ].copy()
        if relevant_df.empty:
            relevant_df = master_df.copy()
        historical_context = format_tickets_for_context(relevant_df)
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "historical_tickets": historical_context,
            "company_code": str(ticket_data["company_code"]),
            "component": str(ticket_data["component"]),
            "summary": str(ticket_data["summary"]),
            "description": str(ticket_data["description"])
        })
        return parse_llm_response(response, default_response)
    except Exception as e:
        print(f"Error in check_duplicate: {str(e)}")
        return default_response

def check_duplicate(master_df, ticket_data):
    """Check if a ticket is duplicate using Vector DB and LLM."""
    # First, check with Vector DB
    vector_db_result = check_duplicate_with_vector_db(ticket_data, master_df)
    if vector_db_result["is_duplicate"]:
        return vector_db_result
    
    # If Vector DB doesn't find a duplicate, check with LLM
    if ENABLE_LLM:
        return check_duplicate_with_llm(master_df, ticket_data, is_fallback=True)
    else:
        return vector_db_result

def check_recently_closed_tickets(ticket_data, master_df):
    """Check if a similar ticket was closed recently (within 30 days)."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "No recent similar tickets found"
    }
    
    try:
        cutoff_date = datetime.now() - timedelta(days=30)
        recent_df = master_df[
            (master_df["closure_date"] >= cutoff_date) &
            ((master_df['company_code'] == ticket_data['company_code']) |
             (master_df['component'] == ticket_data['component']))
        ].copy()
        
        if recent_df.empty:
            return default_response
        
        # Generate embeddings for the new ticket and recently closed tickets
        new_ticket_embedding = generate_embedding(ticket_data["description"])
        if new_ticket_embedding is None:
            return default_response
        
        recent_embeddings = np.array([generate_embedding(desc) for desc in recent_df["description"]])
        recent_embeddings = np.array([emb for emb in recent_embeddings if emb is not None])  # Remove None values
        
        if len(recent_embeddings) == 0:
            return default_response
        
        # Add embeddings to Vector DB index
        index.add(recent_embeddings)
        
        # Search for similar tickets in Vector DB using cosine similarity
        distances, indices = index.search(np.array([new_ticket_embedding]), k=5)
        best_match_index = indices[0][0]
        best_match_distance = distances[0][0]
        
        # Calculate confidence score (cosine similarity is normalized to [0, 1])
        confidence = (1 + best_match_distance) / 2  # Convert inner product to cosine similarity
        
        if confidence >= DUPLICATE_THRESHOLDS["definite_duplicate"]:
            return {
                "is_duplicate": True,
                "original_ticket_id": recent_df.iloc[best_match_index]["ticket_id"],
                "confidence": confidence,
                "reasoning": "High confidence match found in recently closed tickets"
            }
        elif confidence >= DUPLICATE_THRESHOLDS["likely_duplicate"]:
            return {
                "is_duplicate": True,
                "original_ticket_id": recent_df.iloc[best_match_index]["ticket_id"],
                "confidence": confidence,
                "reasoning": "Likely duplicate found in recently closed tickets"
            }
        else:
            return {"is_duplicate": False, "original_ticket_id": None, "confidence": confidence, "reasoning": "No duplicate found in recently closed tickets"}
        
    except Exception as e:
        print(f"Error in check_recently_closed_tickets: {str(e)}")
        return default_response

def is_global_outage(ticket_data, master_df):
    """Check if ticket matches an ongoing global outage issue."""
    try:
        global_outages = master_df[master_df["status"] == "global_outage"].copy()
        
        if global_outages.empty:
            return False
        
        # Generate embeddings for the new ticket and global outage tickets
        new_ticket_embedding = generate_embedding(ticket_data["description"])
        if new_ticket_embedding is None:
            return False
        
        global_outage_embeddings = np.array([generate_embedding(desc) for desc in global_outages["description"]])
        global_outage_embeddings = np.array([emb for emb in global_outage_embeddings if emb is not None])  # Remove None values
        
        if len(global_outage_embeddings) == 0:
            return False
        
        # Add embeddings to Vector DB index
        index.add(global_outage_embeddings)
        
        # Search for similar tickets in Vector DB using cosine similarity
        distances, indices = index.search(np.array([new_ticket_embedding]), k=5)
        best_match_index = indices[0][0]
        best_match_distance = distances[0][0]
        
        # Calculate confidence score (cosine similarity is normalized to [0, 1])
        confidence = (1 + best_match_distance) / 2  # Convert inner product to cosine similarity
        
        if confidence >= DUPLICATE_THRESHOLDS["definite_duplicate"]:
            return True
        else:
            return False
        
    except Exception as e:
        print(f"Error in is_global_outage: {str(e)}")
        return False

# Save the Faiss index to disk
if ENABLE_VECTOR_DB:
    faiss.write_index(index, os.path.join(DATASET_DIR, "faiss_index.index"))

# Rest of the functions (load_master_data, load_new_ticket_data, etc.) remain unchanged
