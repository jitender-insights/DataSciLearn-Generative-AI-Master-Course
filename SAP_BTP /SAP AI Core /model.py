import os
import json
import numpy as np
import faiss
import time
import logging
from datetime import datetime, timedelta
from functools import lru_cache
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ticket_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ticket_analysis")

# Initialize AI model
def initialize_ai_components():
    """Initialize and return AI components with proper error handling."""
    try:
        proxy_client = get_proxy_client("gen-ai-hub")
        llm = init_llm(config.AI_MODEL, proxy_client=proxy_client)
        embeddings = init_embeddings(config.EMBEDDING_MODEL, proxy_client=proxy_client)
        logger.info("AI components initialized successfully")
        return llm, embeddings
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {str(e)}")
        raise RuntimeError(f"AI initialization failed: {str(e)}")

# Initialize AI components
try:
    llm, embeddings = initialize_ai_components()
except Exception as e:
    logger.critical(f"Could not initialize AI components: {str(e)}")
    # In a production system, you might want to use fallback models or retry logic

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

# Configuration validation function
def validate_config():
    """Validate configuration settings to ensure consistency."""
    issues = []
    
    # Check for required configuration parameters
    required_params = [
        "AI_MODEL", "EMBEDDING_MODEL", "ENABLE_VECTOR_DB_DUPLICATE_DETECTION", 
        "ENABLE_LLM_DUPLICATE_DETECTION", "MAX_HISTORICAL_TICKETS", 
        "INDEX_SAVE_LOCATION", "DUPLICATE_THRESHOLDS", "MAX_LLM_RETRIES"
    ]
    
    for param in required_params:
        if not hasattr(config, param):
            issues.append(f"Missing required config parameter: {param}")
    
    # Check specific constraints
    if hasattr(config, "ENABLE_VECTOR_DB_DUPLICATE_DETECTION") and hasattr(config, "ENABLE_LLM_DUPLICATE_DETECTION"):
        if not (config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION or config.ENABLE_LLM_DUPLICATE_DETECTION):
            issues.append("At least one duplicate detection method must be enabled")
    
    if hasattr(config, "DUPLICATE_THRESHOLDS"):
        if not isinstance(config.DUPLICATE_THRESHOLDS, dict):
            issues.append("DUPLICATE_THRESHOLDS must be a dictionary")
        elif ("definite_duplicate" not in config.DUPLICATE_THRESHOLDS or 
              "likely_duplicate" not in config.DUPLICATE_THRESHOLDS):
            issues.append("DUPLICATE_THRESHOLDS must contain 'definite_duplicate' and 'likely_duplicate' keys")
        elif (config.DUPLICATE_THRESHOLDS.get("definite_duplicate", 0) < 
              config.DUPLICATE_THRESHOLDS.get("likely_duplicate", 0)):
            issues.append("'definite_duplicate' threshold must be greater than 'likely_duplicate' threshold")
    
    # Log and return issues
    if issues:
        for issue in issues:
            logger.error(f"Config validation error: {issue}")
        return False, issues
    
    logger.info("Configuration validated successfully")
    return True, []

# Validate configuration on startup
config_valid, config_issues = validate_config()
if not config_valid:
    logger.warning(f"Running with invalid configuration. Issues: {config_issues}")

# Performance metrics tracking
class PerformanceTracker:
    """Track and report performance metrics for different operations."""
    
    def __init__(self):
        self.metrics = {
            "vector_db_searches": [],
            "llm_calls": [],
            "embedding_generation": [],
            "duplicate_checks": {
                "vector_success": 0,
                "vector_failure": 0,
                "llm_success": 0,
                "llm_failure": 0,
                "false_positives": 0,
                "false_negatives": 0
            }
        }
    
    def record_timing(self, operation, elapsed_time):
        """Record timing for an operation."""
        if operation in self.metrics:
            self.metrics[operation].append(elapsed_time)
            
    def record_duplicate_result(self, method, success, false_result=None):
        """Record results of duplicate detection."""
        if method == "vector":
            if success:
                self.metrics["duplicate_checks"]["vector_success"] += 1
            else:
                self.metrics["duplicate_checks"]["vector_failure"] += 1
        elif method == "llm":
            if success:
                self.metrics["duplicate_checks"]["llm_success"] += 1
            else:
                self.metrics["duplicate_checks"]["llm_failure"] += 1
                
        if false_result == "false_positive":
            self.metrics["duplicate_checks"]["false_positives"] += 1
        elif false_result == "false_negative":
            self.metrics["duplicate_checks"]["false_negatives"] += 1
    
    def get_summary(self):
        """Get summary of performance metrics."""
        summary = {}
        
        # Calculate average timings
        for operation in ["vector_db_searches", "llm_calls", "embedding_generation"]:
            if self.metrics[operation]:
                avg_time = sum(self.metrics[operation]) / len(self.metrics[operation])
                summary[f"avg_{operation}_time"] = f"{avg_time:.4f}s"
                summary[f"total_{operation}"] = len(self.metrics[operation])
        
        # Add duplicate check metrics
        summary["duplicate_checks"] = self.metrics["duplicate_checks"]
        
        return summary

# Initialize performance tracker
performance_tracker = PerformanceTracker()

# Enhanced embedding function with caching
@lru_cache(maxsize=1024)
def get_embedding_cached(text):
    """Get embedding vector for text with caching for efficiency."""
    # Remove whitespace from text for caching consistency
    normalized_text = " ".join(text.lower().split())
    
    start_time = time.time()
    try:
        embedding = get_embedding(normalized_text)
        elapsed = time.time() - start_time
        performance_tracker.record_timing("embedding_generation", elapsed)
        logger.debug(f"Generated embedding in {elapsed:.4f}s")
        return embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise

def get_embedding(text, embedding_model=embeddings):
    """Get embedding vector for text using SAP BTP text-embedding-3 model."""
    # Normalize the text by removing excessive whitespace and converting to lowercase
    normalized_text = " ".join(text.lower().split())

    # Call the SAP BTP API to get the embedding
    embedding = embedding_model.embed_documents([normalized_text])[0]

    # Convert the embedding to a numpy array
    return np.array(embedding).astype(np.float32)

# Global vector index manager to avoid recreating indices
class VectorIndexManager:
    """Manage vector indices to avoid recreation for each search."""
    
    def __init__(self):
        self.indices = {}
        self.record_maps = {}
        self.last_updated = {}
    
    def get_index_key(self, df, filter_criteria=None):
        """Generate a cache key for an index based on data and filters."""
        if filter_criteria:
            # Create a sorted string of filter criteria for consistent keys
            filter_str = "_".join(sorted([f"{k}={v}" for k, v in filter_criteria.items()]))
            return f"index_{len(df)}_{hash(filter_str)}"
        return f"index_{len(df)}"
    
    def get_or_create_index(self, df, filter_criteria=None):
        """Get existing index or create a new one."""
        key = self.get_index_key(df, filter_criteria)
        
        # If we have a cached index and the dataframe size hasn't changed
        if key in self.indices and len(df) == self.indices[key]["record_count"]:
            logger.debug(f"Using cached vector index for key {key}")
            return (
                self.indices[key]["index"], 
                self.indices[key]["embeddings"], 
                self.record_maps[key]
            )
        
        # Apply filters if provided
        filtered_df = df
        if filter_criteria:
            for column, value in filter_criteria.items():
                filtered_df = filtered_df[filtered_df[column] == value]
        
        # Convert to records for processing
        records = filtered_df.to_dict(orient="records")
        
        # Create record map to track original indices
        record_map = {i: filtered_df.index[i] for i in range(len(filtered_df))}
        
        # Create new index
        start_time = time.time()
        try:
            index, embeddings = self._create_vector_index(records)
            elapsed = time.time() - start_time
            logger.info(f"Created new vector index in {elapsed:.4f}s with {len(records)} records")
            
            # Cache the new index
            self.indices[key] = {
                "index": index,
                "embeddings": embeddings,
                "record_count": len(df)
            }
            self.record_maps[key] = record_map
            self.last_updated[key] = datetime.now()
            
            return index, embeddings, record_map
        except Exception as e:
            logger.error(f"Failed to create vector index: {str(e)}")
            raise
    
    def _create_vector_index(self, data_records):
        """Create FAISS index from ticket data."""
        # Extract text data for embedding
        texts = []
        for record in data_records:
            # Combine summary and description for embedding
            text = f"{record['summary']} {record['description']}"
            texts.append(text)

        # Generate embeddings
        embeddings = np.array([get_embedding_cached(text) for text in texts]).astype(np.float32)

        # Check if we have any data to index
        if len(embeddings) == 0:
            logger.warning("No data to index, returning empty index")
            embedding_dim = embeddings_model_dimension()  # This should be determined from your model
            index = faiss.IndexFlatL2(embedding_dim)
            return index, embeddings

        # Create FAISS index based on config
        embedding_dim = embeddings.shape[1]

        # Set index type based on config, default to flat index
        if config.INDEX_TYPE == "IndexFlatL2":
            index = faiss.IndexFlatL2(embedding_dim)
        else:
            # Default to HNSW index
            index = faiss.IndexHNSWFlat(embedding_dim, 32)  # 32 connections per node
            index.hnsw.efConstruction = 40
            index.hnsw.efSearch = 16

        # Add vectors to index
        index.add(embeddings)
        
        # Save the index if needed
        if hasattr(config, "SAVE_INDICES") and config.SAVE_INDICES:
            try:
                save_faiss_index(index)
            except Exception as e:
                logger.warning(f"Failed to save index: {str(e)}")

        return index, embeddings

# Initialize vector index manager
vector_index_manager = VectorIndexManager()

def embeddings_model_dimension():
    """Get the dimension of the embedding model."""
    # This is a placeholder - in practice, you should get this from your model config
    return 1536  # Typical dimension for text-embedding-3

def save_faiss_index(index, filename="faiss_index"):
    """Save FAISS index to disk."""
    save_path = os.path.join(config.INDEX_SAVE_LOCATION, filename)

    # Create directory if it doesn't exist
    os.makedirs(config.INDEX_SAVE_LOCATION, exist_ok=True)

    try:
        faiss.write_index(index, save_path)
        logger.info(f"Saved FAISS index to {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {str(e)}")
        raise

def load_faiss_index(filename="faiss_index"):
    """Load FAISS index from disk."""
    load_path = os.path.join(config.INDEX_SAVE_LOCATION, filename)

    if os.path.exists(load_path):
        try:
            index = faiss.read_index(load_path)
            logger.info(f"Loaded FAISS index from {load_path}")
            return index
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {str(e)}")
    
    logger.warning(f"No FAISS index found at {load_path}")
    return None

def search_vector_db(index, query_embedding, k=5):
    """Search vector database for similar tickets."""
    # Reshape query embedding to match FAISS expectations
    query_embedding = query_embedding.reshape(1, -1)

    # Limit k to the number of vectors in the index to avoid out-of-bounds errors
    k = min(k, index.ntotal)

    # If index is empty, return empty results
    if k == 0:
        logger.warning("Empty index or k=0, returning empty results")
        return np.array([]), np.array([])

    # Search the index
    start_time = time.time()
    try:
        distances, indices = index.search(query_embedding, k)
        elapsed = time.time() - start_time
        performance_tracker.record_timing("vector_db_searches", elapsed)
        logger.debug(f"Vector search completed in {elapsed:.4f}s")
        return distances[0], indices[0]  # Return first result's distances and indices
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        raise

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
            logger.warning(f"Missing required fields in LLM response: {response_text}")
            return default_response

        return parsed
    except Exception as e:
        logger.error(f"Error parsing LLM response: {str(e)}")
        logger.debug(f"Raw response: {response_text}")
        return default_response

def validate_ticket_data(ticket_data):
    """Pre-validate ticket data before processing."""
    # Check for required fields
    for field in config.PRE_VALIDATION_FIELDS:
        if not ticket_data.get(field):
            logger.warning(f"Ticket validation failed: Missing required field: {field}")
            return False, f"Missing required field: {field}"

    return True, "Validation successful"

# Centralized pre-filtering function
def prefilter_tickets(master_df, ticket_data):
    """Centralized function to filter tickets by company code and component."""
    if not all(field in ticket_data for field in ['company_code', 'component']):
        logger.warning("Cannot prefilter: Missing required fields in ticket data")
        return master_df.copy()
    
    try:
        filtered_df = master_df[
            (master_df['company_code'] == ticket_data['company_code']) &
            (master_df['component'] == ticket_data['component'])
        ].copy()
        
        logger.info(f"Prefiltering reduced dataset from {len(master_df)} to {len(filtered_df)} records")
        return filtered_df
    except Exception as e:
        logger.error(f"Error during prefiltering: {str(e)}")
        return master_df.copy()

def check_duplicate(master_df, ticket_data):
    """Check if a ticket is duplicate using direct LLM analysis or Vector DB based on config."""
    start_time = time.time()
    
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }

    # Pre-validation: Only process tickets with valid data
    if hasattr(config, "ENABLE_PRE_FILTERING") and config.ENABLE_PRE_FILTERING:
        is_valid, validation_message = validate_ticket_data(ticket_data)
        if not is_valid:
            logger.warning(f"Ticket validation failed: {validation_message}")
            return {
                "is_duplicate": False,
                "original_ticket_id": None,
                "confidence": 0.0,
                "reasoning": validation_message
            }

    try:
        # Centralized pre-filtering
        relevant_df = prefilter_tickets(master_df, ticket_data)
        
        # Skip further processing if no relevant tickets found
        if relevant_df.empty:
            logger.info("No relevant tickets found after prefiltering")
            return {
                "is_duplicate": False,
                "original_ticket_id": None,
                "confidence": 0.0,
                "reasoning": "No matching company code and component found"
            }
            
        # First try Vector DB approach if enabled
        if hasattr(config, "ENABLE_VECTOR_DB_DUPLICATE_DETECTION") and config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION:
            logger.info("Attempting vector-based duplicate detection")
            
            try:
                # Get or create vector index for relevant tickets
                filter_criteria = {
                    'company_code': ticket_data['company_code'],
                    'component': ticket_data['component']
                }
                vector_index, embeddings, record_map = vector_index_manager.get_or_create_index(
                    master_df, filter_criteria
                )

                # Generate embedding for new ticket
                query_text = f"{ticket_data['summary']} {ticket_data['description']}"
                query_embedding = get_embedding_cached(query_text)

                # Search for similar tickets
                distances, indices = search_vector_db(vector_index, query_embedding, k=min(5, len(relevant_df)))

                # Skip if no results returned
                if indices.size == 0:
                    logger.info("No similar tickets found in vector search")
                    return default_response

                # Calculate similarities from L2 distances
                epsilon = 1e-10
                max_distance = np.max(distances) if distances.size > 0 else 1.0
                max_distance = max(max_distance, epsilon)  # Avoid division by zero

                similarities = []
                for dist in distances:
                    # Handle zero distance as perfect similarity
                    if dist == 0:
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

                # Get the matching ticket using the record map
                if best_idx >= 0 and best_idx < len(indices) and indices[best_idx] < len(record_map):
                    # Map from vector index to original dataframe index
                    vector_idx = indices[best_idx]
                    if vector_idx in record_map:
                        original_idx = record_map[vector_idx]
                        
                        # Access the original dataframe using the mapped index
                        if original_idx in master_df.index:
                            matching_ticket = master_df.loc[original_idx]

                            # Determine duplicate status based on thresholds
                            if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate']:
                                result = {
                                    "is_duplicate": True,
                                    "original_ticket_id": matching_ticket['ticket_id'],
                                    "confidence": best_similarity,
                                    "reasoning": f"Vector similarity score: {best_similarity:.4f} - Definite duplicate"
                                }
                                performance_tracker.record_duplicate_result("vector", True)
                                return result
                            elif best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate']:
                                result = {
                                    "is_duplicate": True,
                                    "original_ticket_id": matching_ticket['ticket_id'],
                                    "confidence": best_similarity,
                                    "reasoning": f"Vector similarity score: {best_similarity:.4f} - Likely duplicate"
                                }
                                performance_tracker.record_duplicate_result("vector", True)
                                return result
                
                logger.info(f"Vector search found tickets but similarity ({best_similarity:.4f}) below threshold")
                performance_tracker.record_duplicate_result("vector", False)
                
            except Exception as e:
                logger.error(f"Vector-based duplicate detection failed: {str(e)}")
                performance_tracker.record_duplicate_result("vector", False)
                # Continue to LLM-based approach as fallback

        # Fall back to LLM approach if Vector DB not enabled or didn't find a match
        if hasattr(config, "ENABLE_LLM_DUPLICATE_DETECTION") and config.ENABLE_LLM_DUPLICATE_DETECTION:
            logger.info("Attempting LLM-based duplicate detection")
            
            historical_context = format_tickets_for_context(relevant_df)

            # Prepare and execute the chain
            prompt = ChatPromptTemplate.from_template(DUPLICATE_CHECK_TEMPLATE)
            chain = prompt | llm | StrOutputParser()

            # Implement retry logic
            retries = 0
            while retries < config.MAX_LLM_RETRIES:
                try:
                    start_llm_time = time.time()
                    response = chain.invoke({
                        "historical_tickets": historical_context,
                        "company_code": str(ticket_data["company_code"]),
                        "component": str(ticket_data["component"]),
                        "summary": str(ticket_data["summary"]),
                        "description": str(ticket_data["description"])
                    })
                    llm_elapsed = time.time() - start_llm_time
                    performance_tracker.record_timing("llm_calls", llm_elapsed)

                    result = parse_llm_response(response, default_response)
                    if result != default_response:
                        if result.get("is_duplicate", False):
                            performance_tracker.record_duplicate_result("llm", True)
                        else:
                            performance_tracker.record_duplicate_result("llm", False)
                        return result

                    retries += 1
                    logger.warning(f"LLM response parsing failed, retry {retries}/{config.MAX_LLM_RETRIES}")
                except Exception as e:
                    logger.error(f"LLM retry {retries+1}/{config.MAX_LLM_RETRIES} failed: {str(e)}")
                    retries += 1
                    time.sleep(1)  # Add backoff

            # If we get here, all retries failed
            logger.error("All LLM retries failed")
            performance_tracker.record_duplicate_result("llm", False)
            return default_response

        # If neither Vector DB nor LLM analysis is enabled
        logger.warning("No duplicate detection methods enabled")
        return default_response

    except Exception as e:
        logger.error(f"Error in check_duplicate: {str(e)}")
        return default_response
    finally:
        total_elapsed = time.time() - start_time
        logger.info(f"Duplicate check completed in {total_elapsed:.4f}s")

def check_global_outage(ticket_data, global_outages_df):
    """Check if a ticket indicates a global outage."""
    default_response = {
        "is_global_outage": False,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }
    
    # Validate ticket data
    is_valid, validation_message = validate_ticket_data(ticket_data)
    if not is_valid:
        logger.warning(f"Ticket validation failed for global outage check: {validation_message}")
        return default_response
        
    try:
        # Format global outages for context
        historical_context = format_tickets_for_context(global_outages_df)
        
        # Prepare and execute the chain
        prompt = ChatPromptTemplate.from_template(GLOBAL_OUTAGE_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        
        # Implement retry logic
        retries = 0
        while retries < config.MAX_LLM_RETRIES:
            try:
                start_time = time.time()
                response = chain.invoke({
                    "global_outages": historical_context,
                    "summary": str(ticket_data["summary"]),
                    "description": str(ticket_data["description"])
                })
                elapsed = time.time() - start_time
                performance_tracker.record_timing("llm_calls", elapsed)
                
                result = parse_llm_response(response, default_response)
                if result != default_response:
                    return result
                    
                retries += 1
                logger.warning(f"Global outage LLM response parsing failed, retry {retries}/{config.MAX_LLM_RETRIES}")
            except Exception as e:
                logger.error(f"Global outage LLM retry {retries+1}/{config.MAX_LLM_RETRIES} failed: {str(e)}")
                retries += 1
                time.sleep(1)  # Add backoff
                
        return default_response
    except Exception as e:
        logger.error(f"Error in check_global_outage: {str(e)}")
        return default_response

def check_recently_closed_tickets(ticket_data, master_df):
    """Check if a similar ticket was closed recently (within configured days)."""
    start_time = time.time()
    
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "No recent similar tickets found"
    }

    # Pre-validation
    is_valid, validation_message = validate_ticket_data(ticket_data)
    if not is_valid:
        logger.warning(f"Ticket validation failed for recently closed check: {validation_message}")
        return default_response

    try:
        # Pre-filter by company code, component, and recent closure date
        cutoff_date = datetime.now() - timedelta(days=config.RECENT_TICKETS_DAYS)
        filter_criteria = {
            'company_code': ticket_data['company_code'],
            'component': ticket_data['component']
        }
        
        # Apply date filter separately since it's not an equality filter
        recent_df = prefilter_tickets(master_df, ticket_data)
        recent_df = recent_df[recent_df["closure_date"] >= cutoff_date].copy()

        if recent_df.empty:
            logger.info("No recently closed tickets found")
            return default_response

        # Decide which approach to use based on configuration
        if hasattr(config, "ENABLE_VECTOR_DB_DUPLICATE_DETECTION") and config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION:
            logger.info("Attempting vector-based recently closed ticket search")
            
            try:
                # Create a combined filter criteria
                combined_filter = {
                    'company_code': ticket_data['company_code'],
                    'component': ticket_data['component'],
                    'recent': True  # Add a flag for recent tickets
                }
                
                # Get or create vector index for recent tickets
                vector_index, embeddings, record_map = vector_index_manager.get_or_create_index(
                    recent_df, combined_filter
                )

                # Generate embedding for new ticket
                query_text = f"{ticket_data['summary']} {ticket_data['description']}"
                query_embedding = get_embedding_cached(query_text)

                # Search in vector database - limit k to the size of recent records
                distances, indices = search_vector_db(vector_index, query_embedding, k=min(5, len(recent_df)))

                # Skip if no results
                if indices.size == 0:
                    logger.info("No similar recently closed tickets found in vector search")
                    return default_response

                # Calculate similarities from L2 distances
                epsilon = 1e-10
                max_distance = np.max(distances) if distances.size > 0 else 1.0
                max_distance = max(max_distance, epsilon)  # Avoid division by zero

                similarities = []
                for dist in distances:
                    # Handle zero distance as perfect similarity
                    if dist == 0:
                        similarities.append(1.0)
                    else:
                        norm_dist = dist / max_distance
                        similarities.append(1.0 - norm_dist)

                # Find the best match
                if not similarities:
                    return default_
