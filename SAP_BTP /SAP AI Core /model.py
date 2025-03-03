# Update the check_duplicate function
def check_duplicate(master_df, ticket_data):
    """Check if a ticket is duplicate using direct LLM analysis or Vector DB based on config."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "Unable to perform analysis"
    }

    # Apply the new LLM-based content validation
    is_valid, validation_message, quality_score = validate_ticket_content_quality(ticket_data, llm)
    if not is_valid:
        return {
            "is_duplicate": False,
            "original_ticket_id": None,
            "confidence": 0.0,
            "reasoning": validation_message
        }

    # Existing pre-validation (keep as is)
    if config.ENABLE_PRE_FILTERING:
        is_valid, validation_message = validate_ticket_data(ticket_data)
        if not is_valid:
            return {
                "is_duplicate": False,
                "original_ticket_id": None,
                "confidence": 0.0,
                "reasoning": validation_message
            }

    # Rest of your existing function remains unchanged
    try:
        # First try Vector DB approach if enabled
        # ... (existing code)
    except Exception as e:
        print(f"Error in check_duplicate: {str(e)}")
        return default_response

# Similarly update check_recently_closed_tickets
def check_recently_closed_tickets(ticket_data, master_df):
    """Check if a similar ticket was closed recently (within configured days)."""
    default_response = {
        "is_duplicate": False,
        "original_ticket_id": None,
        "confidence": 0.0,
        "reasoning": "No recent similar tickets found"
    }

    # Apply the new LLM-based content validation
    is_valid, validation_message, quality_score = validate_ticket_content_quality(ticket_data, llm)
    if not is_valid:
        return {
            "is_duplicate": False,
            "original_ticket_id": None,
            "confidence": 0.0,
            "reasoning": validation_message
        }

    # Existing pre-validation
    is_valid, validation_message = validate_ticket_data(ticket_data)
    if not is_valid:
        return default_response

    # Rest of your existing function remains unchanged
    try:
        # ... (existing code)
    except Exception as e:
        print(f"Error in check_recently_closed_tickets: {str(e)}")
        return default_response

# Add to config.py
# Content quality validation config
MIN_SUMMARY_LENGTH = 10  # Minimum characters for fallback validation
MIN_DESCRIPTION_LENGTH = 30  # Minimum characters for fallback validation
IDEAL_SUMMARY_WORDS = 10  # For fallback quality score calculation
IDEAL_DESCRIPTION_WORDS = 50  # For fallback quality score calculation
ENABLE_LLM_CONTENT_VALIDATION = True  # Toggle for LLM-based validation
