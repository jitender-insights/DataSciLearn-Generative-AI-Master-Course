def check_duplicate(master_df, ticket_data):
    """Check if a ticket is duplicate using direct LLM analysis or Vector DB based on config."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }
    decision_start_time = time.time()
    method_used = "none"
    final_result = None
    best_similarity = 0.0

    # Pre-validation
    if config.ENABLE_PRE_FILTERING:
        is_valid, validation_message = validate_ticket_data(ticket_data)
        if not is_valid:
            metrics.log_decision(
                is_duplicate=False,
                confidence=0.0,
                method="pre-filter-fail"
            )
            return {
                "is_duplicate": False,
                "original_ticket_id": None,
                "confidence": 0.0,
                "reasoning": validation_message
            }

    try:
        # Vector DB Approach
        if config.ENABLE_VECTOR_DB_DUPLICATE_DETECTION:
            vector_search_start = time.time()
            relevant_df = master_df[
                (master_df['company_code'] == ticket_data['company_code']) &
                (master_df['component'] == ticket_data['component'])
            ].copy()

            if not relevant_df.empty:
                relevant_records = relevant_df.to_dict(orient="records")
                vector_index, _ = create_vector_index(relevant_records)
                
                query_text = f"{ticket_data['summary']} {ticket_data['description']}"
                query_embedding = get_embedding(query_text)
                
                distances, indices = search_vector_db(vector_index, query_embedding, k=min(5, len(relevant_records)))
                
                if indices.size > 0:
                    # Similarity calculation
                    max_distance = np.max(distances) if distances.size > 0 else 1.0
                    epsilon = 1e-10
                    max_distance = max(max_distance, epsilon)
                    
                    similarities = []
                    for dist in distances:
                        if dist == 0:
                            similarities.append(1.0)
                        else:
                            norm_dist = dist / max_distance
                            similarities.append(1.0 - norm_dist)
                    
                    best_idx = np.argmax(similarities)
                    best_similarity = similarities[best_idx]
                    metrics.log_similarity(best_similarity, False)  # Temp log before final decision

                    if best_idx >= 0 and indices[best_idx] < len(relevant_df):
                        matching_ticket = relevant_df.iloc[indices[best_idx]]
                        
                        # Threshold checks with metrics
                        if best_similarity >= config.DUPLICATE_THRESHOLDS['definite_duplicate']:
                            final_result = {
                                "is_duplicate": True,
                                "original_ticket_id": matching_ticket['ticket_id'],
                                "confidence": best_similarity,
                                "reasoning": f"Vector similarity: {best_similarity:.4f} - Definite duplicate"
                            }
                            metrics.log_threshold_check('definite', best_similarity)
                            method_used = "vector"
                        elif best_similarity >= config.DUPLICATE_THRESHOLDS['likely_duplicate']:
                            final_result = {
                                "is_duplicate": True,
                                "original_ticket_id": matching_ticket['ticket_id'],
                                "confidence": best_similarity,
                                "reasoning": f"Vector similarity: {best_similarity:.4f} - Likely duplicate"
                            }
                            metrics.log_threshold_check('likely', best_similarity)
                            method_used = "vector"
                        else:
                            metrics.log_threshold_check('below_threshold', best_similarity)

            metrics.log_vector_search(
                k=min(5, len(relevant_records)),
                num_candidates=len(relevant_records),
                search_time=time.time() - vector_search_start
            )

        # LLM Fallback
        if not final_result and config.ENABLE_LLM_DUPLICATE_DETECTION:
            llm_start_time = time.time()
            retries = 0
            llm_success = False
            
            relevant_df = master_df[
                (master_df['company_code'] == ticket_data['company_code']) &
                (master_df['component'] == ticket_data['component'])
            ].copy()
            
            if not relevant_df.empty:
                historical_context = format_tickets_for_context(relevant_df)
                prompt = ChatPromptTemplate.from_template(DUPLICATE_CHECK_TEMPLATE)
                chain = prompt | llm | StrOutputParser()
                
                while retries < config.MAX_LLM_RETRIES and not llm_success:
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
                            final_result = result
                            method_used = "llm"
                            llm_success = True
                            metrics.log_similarity(result['confidence'], result['is_duplicate'])
                    except Exception as e:
                        retries += 1
                    
                metrics.log_llm_call(
                    success=llm_success,
                    retries=retries,
                    response_time=time.time() - llm_start_time
                )

        # Final decision logging
        if final_result:
            metrics.log_decision(
                is_duplicate=final_result['is_duplicate'],
                confidence=final_result['confidence'],
                method=method_used
            )
        else:
            metrics.log_decision(
                is_duplicate=False,
                confidence=0.0,
                method="none"
            )

        return final_result if final_result else default_response

    except Exception as e:
        print(f"Error in check_duplicate: {str(e)}")
        metrics.log_decision(
            is_duplicate=False,
            confidence=0.0,
            method="error"
        )
        return default_response
    finally:
        metrics.log_decision_time(time.time() - decision_start_time)
