def save_chroma_store(vector_store, metadata=None, collection_name="ticket_collection"):
    """Persist ChromaDB vector store to disk."""
    # Create directory if it doesn't exist
    os.makedirs(config.INDEX_SAVE_LOCATION, exist_ok=True)
    
    # Persist the vector store
    vector_store.persist()
    
    # Optionally save metadata about the index
    if metadata:
        metadata_path = os.path.join(config.INDEX_SAVE_LOCATION, f"{collection_name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    return os.path.join(config.INDEX_SAVE_LOCATION, collection_name)

def load_chroma_store(collection_name="ticket_collection"):
    """Load ChromaDB vector store from disk."""
    chroma_path = os.path.join(config.INDEX_SAVE_LOCATION, collection_name)
    
    # Check if the directory exists
    if not os.path.exists(chroma_path):
        return None
    
    # Load the vector store
    try:
        vector_store = Chroma(
            persist_directory=chroma_path,
            embedding_function=embeddings
        )
        return vector_store
    except Exception as e:
        print(f"Error loading Chroma vector store: {str(e)}")
        return None
def create_vector_index(data_records):
    """Create ChromaDB index from ticket data."""
    # First check if we have a saved collection
    vector_store = load_chroma_store()
    
    # Extract text data for embedding
    texts = []
    metadatas = []
    ids = []
    
    for i, record in enumerate(data_records):
        # Combine summary and description for embedding
        text = f"{record['summary']} {record['description']}"
        texts.append(text)
        
        # Prepare metadata - store company_code and component
        metadata = {
            "ticket_id": record.get("ticket_id", f"ticket_{i}"),
            "company_code": record.get("company_code", ""),
            "component": record.get("component", ""),
            "summary": record.get("summary", ""),
            "description": record.get("description", "")
        }
        metadatas.append(metadata)
        
        # Create a unique ID
        ids.append(record.get("ticket_id", f"ticket_{i}"))
    
    # If we have data to index
    if texts:
        # Create a new vector store if one doesn't exist
        if vector_store is None:
            vector_store = Chroma.from_texts(
                texts=texts,
                embedding=embeddings,
                metadatas=metadatas,
                ids=ids,
                persist_directory=os.path.join(config.INDEX_SAVE_LOCATION, "ticket_collection")
            )
        else:
            # Add documents to existing store
            vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
        
        # Persist the vector store
        vector_store.persist()
        
        # For compatibility with existing code, also return embeddings as numpy array
        embeddings_array = np.array([get_embedding(text) for text in texts]).astype(np.float32)
        
        return vector_store, embeddings_array
    
    # If no data, return empty vector store and empty embeddings array
    return vector_store if vector_store else None, np.array([])

def search_vector_db(vector_store, query_embedding, k=5):
    """Search ChromaDB vector store for similar tickets."""
    if vector_store is None:
        return np.array([]), np.array([])
    
    # Convert query embedding to text for ChromaDB
    query_text = query_embedding_to_text(query_embedding)
    
    # Perform search
    results = vector_store.similarity_search_with_relevance_scores(
        query=query_text,
        k=k
    )
    
    # If no results, return empty arrays
    if not results:
        return np.array([]), np.array([])
    
    # Extract distances and indices
    distances = []
    indices = []
    
    for i, (doc, score) in enumerate(results):
        # Convert similarity score to distance (1 - similarity)
        # Note: ChromaDB returns similarity scores in [0,1], we convert to distance
        distance = 1.0 - score
        distances.append(distance)
        
        # Get the index of this document
        # This is a bit tricky as ChromaDB doesn't return indices directly
        # We'll use the position in the results as a proxy
        indices.append(i)
    
    return np.array(distances), np.array(indices)

def query_embedding_to_text(embedding):
    """Convert an embedding back to queryable text.
    This is a workaround since ChromaDB uses text queries not raw embeddings."""
    # Option 1: Use a placeholder text and rely on the embedding function
    return "query_placeholder"
    
    # Option 2 (if available): Reverse lookup the most similar document
    # This would require keeping a cache of (text -> embedding) mappings

def check_duplicate(master_df, ticket_data):
    """Check if a ticket is duplicate using direct LLM analysis or Vector DB based on config."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }

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
                vector_store, embeddings = create_vector_index(relevant_records)

                # Generate embedding for new ticket
                query_text = f"{ticket_data['summary']} {ticket_data['description']}"
                
                # Use ChromaDB's native similarity search instead of raw embeddings
                results = vector_store.similarity_search_with_relevance_scores(
                    query=query_text,
                    k=min(5, len(relevant_records)),
                    filter={
                        "company_code": ticket_data['company_code'],
                        "component": ticket_data['component']
                    }
                )

                # Skip if no results returned
                if not results:
                    return default_response

                # Process similarity scores
                best_similarity = 0
                best_match = None
                
                for doc, similarity in results:
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = doc

                # Determine duplicate status based on thresholds
                if best_match and best_similarity:
                    # Get ticket_id from metadata
                    matching_ticket_id = best_match.metadata.get('ticket_id')
                    
                    if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate']:
                        return {
                            "is_duplicate": True,
                            "original_ticket_id": matching_ticket_id,
                            "confidence": best_similarity,
                            "reasoning": f"Vector similarity score: {best_similarity:.4f} - Definite duplicate"
                        }
                    elif best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate']:
                        return {
                            "is_duplicate": True,
                            "original_ticket_id": matching_ticket_id,
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
            vector_store, _ = create_vector_index(recent_records)
            
            if vector_store is None:
                return default_response

            # Generate query for new ticket
            query_text = f"{ticket_data['summary']} {ticket_data['description']}"
            
            # Search with filters
            results = vector_store.similarity_search_with_relevance_scores(
                query=query_text,
                k=min(5, len(recent_records)),
                filter={
                    "company_code": ticket_data['company_code'],
                    "component": ticket_data['component']
                }
            )
            
            # Skip if no results
            if not results:
                return default_response

            # Find the best match
            best_similarity = 0
            best_match = None
            
            for doc, similarity in results:
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = doc

            # Check against thresholds
            if best_match and best_similarity:
                matching_ticket_id = best_match.metadata.get('ticket_id')
                
                if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate'] or best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate']:
                    return {
                        "is_duplicate": True,
                        "original_ticket_id": matching_ticket_id,
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
