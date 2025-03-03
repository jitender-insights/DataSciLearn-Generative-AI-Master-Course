import re
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import json

# Define the prompt template for ticket content validation
TICKET_VALIDATION_TEMPLATE = """
You are an expert support ticket analyst. Your task is to evaluate if a support ticket contains sufficient information
to be properly analyzed for potential duplicates.

Analyze the following ticket:

Company Code: {company_code}
Component: {component}
Summary: {summary}
Description: {description}

Evaluate the ticket based on the following criteria:
1. Does the summary clearly indicate what the issue is about?
2. Does the description provide specific details about the problem?
3. Is there sufficient information to identify similar tickets?
4. Are there specific error messages, steps to reproduce, or technical details?

Return your analysis in the following JSON format:
```json
{
  "is_valid": true or false,
  "confidence_score": a value between 0.0 and 1.0 representing your confidence in the quality of this ticket,
  "reasoning": "brief explanation of your decision",
  "missing_information": ["list of information that would improve the ticket"]
}
```

Your reasoning should be concise and clear.
"""

def validate_ticket_content_quality(ticket_data, llm):
    """
    Validate ticket content quality before duplication detection using LLM.
    
    Args:
        ticket_data (dict): Ticket data containing summary, description, etc.
        llm: LLM model instance for content analysis
        
    Returns:
        tuple: (is_valid, validation_message, confidence_score)
    """
    # Default response for invalid tickets
    default_validation = (False, "Insufficient information for duplicate detection", 0.0)
    
    # Basic validation for required fields
    required_fields = ['summary', 'description', 'company_code', 'component']
    for field in required_fields:
        if field not in ticket_data or not ticket_data[field]:
            return False, f"Missing required field: {field}", 0.0
    
    # Basic length checks before sending to LLM (to avoid wasting API calls on obviously empty tickets)
    summary = str(ticket_data['summary']).strip()
    description = str(ticket_data['description']).strip()
    
    # Absolute minimum length check
    if len(summary) < 3 or len(description) < 10:
        return False, "Summary or description is too short for analysis", 0.0
    
    # Use LLM for deeper content analysis
    try:
        # Create prompt with ticket data
        prompt = ChatPromptTemplate.from_template(TICKET_VALIDATION_TEMPLATE)
        chain = prompt | llm | StrOutputParser()
        
        # Implement retry logic
        retries = 0
        while retries < config.MAX_LLM_RETRIES:
            try:
                response = chain.invoke({
                    "company_code": str(ticket_data["company_code"]),
                    "component": str(ticket_data["component"]),
                    "summary": str(ticket_data["summary"]),
                    "description": str(ticket_data["description"])
                })
                
                # Parse LLM response
                validation_result = parse_validation_response(response, default_validation)
                
                # Return the result
                return (
                    validation_result.get("is_valid", False),
                    validation_result.get("reasoning", "Content validation failed"),
                    validation_result.get("confidence_score", 0.0)
                )
                
            except Exception as e:
                print(f"LLM validation retry {retries+1}/{config.MAX_LLM_RETRIES} failed: {str(e)}")
                retries += 1
        
        # If all retries failed, fall back to basic validation
        return fallback_validation(ticket_data)
        
    except Exception as e:
        print(f"Error in LLM ticket validation: {str(e)}")
        # Fall back to basic validation if LLM analysis fails
        return fallback_validation(ticket_data)

def parse_validation_response(response_text, default_validation):
    """
    Parse LLM response for ticket validation.
    
    Args:
        response_text (str): LLM response text
        default_validation (tuple): Default validation result to use if parsing fails
        
    Returns:
        dict: Parsed validation result
    """
    try:
        # Clean up common JSON formatting issues
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "")
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "")
        
        # Parse JSON
        parsed = json.loads(response_text)
        
        # Validate required fields
        if "is_valid" not in parsed or "confidence_score" not in parsed:
            return {
                "is_valid": default_validation[0],
                "reasoning": default_validation[1],
                "confidence_score": default_validation[2]
            }
        
        return parsed
    except Exception as e:
        print(f"Error parsing LLM validation response: {str(e)}")
        print(f"Raw response: {response_text}")
        return {
            "is_valid": default_validation[0],
            "reasoning": default_validation[1],
            "confidence_score": default_validation[2]
        }

def fallback_validation(ticket_data):
    """
    Perform basic validation if LLM analysis fails.
    
    Args:
        ticket_data (dict): Ticket data to validate
        
    Returns:
        tuple: (is_valid, validation_message, confidence_score)
    """
    # Extract fields for analysis
    summary = str(ticket_data['summary']).strip()
    description = str(ticket_data['description']).strip()
    
    # Check for minimum content length
    if len(summary) < config.MIN_SUMMARY_LENGTH:
        return False, f"Summary too short (minimum {config.MIN_SUMMARY_LENGTH} characters required)", 0.0
    
    if len(description) < config.MIN_DESCRIPTION_LENGTH:
        return False, f"Description too short (minimum {config.MIN_DESCRIPTION_LENGTH} characters required)", 0.0
    
    # Check for information content (not just placeholders)
    placeholder_patterns = [
        r'test', r'testing', r'n/a', r'not applicable', r'none', r'placeholder',
        r'to be added', r'to be determined', r'tbd', r'[.]+', r'[-]+', r'[*]+'
    ]
    
    # Check if summary or description only contains placeholder text
    for pattern in placeholder_patterns:
        if re.match(f'^{pattern}$', summary.lower()) or re.match(f'^{pattern}$', description.lower()):
            return False, "Placeholder text detected in summary or description", 0.0
    
    # Calculate content quality score (basic implementation)
    word_count_summary = len(summary.split())
    word_count_desc = len(description.split())
    
    # Calculate simple content quality score
    quality_score = min(1.0, (word_count_summary / config.IDEAL_SUMMARY_WORDS + 
                             word_count_desc / config.IDEAL_DESCRIPTION_WORDS) / 2)
    
    # Use a relatively low threshold for fallback validation
    if quality_score < 0.3:
        return False, "Insufficient information in ticket description", quality_score
        
    return True, "Basic ticket validation passed", quality_score
