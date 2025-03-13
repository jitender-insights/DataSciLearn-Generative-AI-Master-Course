@app.route('/process_ticket', methods=['POST'])
def process_ticket():
    """
    Main endpoint to process the ticket:
      1. Checks for duplicates.
      2. If duplicate, returns subtask info + debug info for duplicates.
      3. Otherwise, classifies with different confidence messages:
         - High confidence  (>= config.CLASSIFICATION_HIGH_CONF_THRESHOLD)
         - Moderate         (>= config.CLASSIFICATION_MODERATE_CONF_THRESHOLD and < high threshold)
         - Low              (< config.CLASSIFICATION_MODERATE_CONF_THRESHOLD)
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in process_ticket")
            return jsonify({"error": "No JSON data received"}), 400
        summary = data.get("Summary", data.get("summary", "")).strip()
        description = data.get("Description", data.get("description", "")).strip()
        component = data.get("Component", data.get("component", "")).strip()
        company_code = data.get("Company_Code", data.get("company_code", "")).strip()

        # Mandatory fields
        if not summary:
            logging.error("Summary is mandatory and cannot be blank.")
            return jsonify({"error": "Summary is mandatory and cannot be blank."}), 400
        if not description:
            logging.error("Description is mandatory and cannot be blank.")
            return jsonify({"error": "Description is mandatory and cannot be blank."}), 400
        if not component:
            logging.error("Component is mandatory and cannot be blank.")
            return jsonify({"error": "Component is mandatory and cannot be blank."}), 400
        if not company_code:
            logging.error("Company Code is mandatory and cannot be blank.")
            return jsonify({"error": "Company Code is mandatory and cannot be blank."}), 400

        logging.debug(
            f"Process Ticket - Received: Summary='{summary}', "
            f"Description='{description}', Company_Code='{company_code}', Component='{component}'"
        )

        from analysis import analyze_ticket_data  # Adjust to your code
        is_valid, response_message = analyze_ticket_data(summary, description, component, company_code)
        if not is_valid:
            logging.warning("Insufficient ticket data")
            return jsonify({"error": response_message}), 400
        # 1. Check for duplicates
        duplicate_result = check_duplicate_ticket(
            {
                "Summary": summary,
                "Description": description,
                "Company_Code": company_code,
                "Component": component
            },
            vectorstore,
            embeddings
        )
        logging.debug(f"Duplicate detection result: {duplicate_result}")

        # Extract duplicate metadata
        duplicate_category = duplicate_result.get("duplicate_category", "")
        similarity = duplicate_result.get("similarity", 0.0)
        reasoning = duplicate_result.get("reasoning", "")
        is_duplicate_flag = duplicate_result.get("is_duplicate", False)
        original_ticket_id = duplicate_result.get("original_ticket_id", "")

        # Build a base debug dict that we will return in all branches
        duplicate_debug = {
            "duplicate_category": duplicate_category,
            "is_duplicate": str(is_duplicate_flag),
            "original_ticket_id": str(original_ticket_id),
            "similarity": str(similarity),
            "reasoning": reasoning if reasoning else ""
        }
        # --------------------------------------------------------------------
        # A) Strong duplicate: "Likely duplicate" or "Combined duplicate"
        # --------------------------------------------------------------------
        if is_duplicate_flag and duplicate_category in ["Likely duplicate", "Combined duplicate"]:
            subtask = create_subtask(original_ticket_id, {"Description": description})
            logging.info(f"Duplicate detected for ticket ID {original_ticket_id}")
            return jsonify({
                "message": (
                    "Duplicate ticket detected, Creating subtask in JIRA.\n"
                    "Changing Status to 'DUPLICATE_VERIFICATION_NEEDED'"
                ),
                "subtask": subtask,
                "duplicate_debug": duplicate_debug
            })
        # --------------------------------------------------------------------
        # B) May be duplicate (moderate similarity) – require agent decision
        # --------------------------------------------------------------------
        elif duplicate_category.lower() == "may be duplicate":
            # Check for agent decision for duplicate handling
            duplicate_decision = data.get("duplicate_decision", "").strip().lower()
            if duplicate_decision in ["accept duplicate", "accept_duplicate"]:
                subtask = create_subtask(original_ticket_id, {"Description": description})
                message = "Ticket flagged as 'May be duplicate' and agent accepted duplicate. Creating subtask in JIRA."
                logging.info(message)
                return jsonify({
                    "message": message,
                    "subtask": subtask,
                    "duplicate_debug": duplicate_debug
                })
            elif duplicate_decision in ["reject duplicate", "reject_duplicate"]:
                # Agent has chosen to reject duplicate handling so proceed to classification
                logging.info("Ticket flagged as 'May be duplicate' but agent rejected duplicate. Proceeding with classification.")
                # Optionally update duplicate_debug with extra reasoning:
                duplicate_debug["reasoning"] = f"May be duplicate: {reasoning}" if reasoning else "Moderate similarity observed. Further evaluation needed."
                confidence_message = "Ticket is flagged as 'May be duplicate' but agent rejected duplicate. Proceeding with classification."
                # Fall-through to classification logic below:
            else:
                message = ("Ticket flagged as 'May be duplicate'. Awaiting agent decision. "
                           "Please provide 'duplicate_decision' as 'accept duplicate' or 'reject duplicate'.")
                logging.info(message)
                return jsonify({"message": message, "duplicate_debug": duplicate_debug})
        # --------------------------------------------------------------------
        # C) Not a duplicate or classification after rejecting duplicate in may be branch
        # --------------------------------------------------------------------
        logging.info("No strong duplicate detected. Proceeding with classification & confidence logic.")
        if not reasoning:
            duplicate_debug["reasoning"] = "No strong similarity found. Ticket is not a duplicate."
        retrieved_tickets = retrieve_and_rerank_tickets(vectorstore, summary, description, k=5)
        classification = generate_response_with_prompt(llm, retrieved_tickets, summary, description)
        # Evaluate final confidence from best ticket's Combined_Score
        confidence_message = ""
        default_status = "Manual_Classification_Required"
        if retrieved_tickets:
            best_ticket = retrieved_tickets[0]
            best_score = best_ticket["Combined_Score"]
            if best_score >= config.CLASSIFICATION_HIGH_CONF_THRESHOLD:
                confidence_message = (
                    "Ticket is not duplicate. High confidence match. "
                    "Classifying ticket now and updating JIRA status to 'classified'."
                )
                default_status = "classified"
            elif best_score >= config.CLASSIFICATION_MODERATE_CONF_THRESHOLD:
                # New logic: require agent decision for moderate similarity classification
                classification_decision = data.get("classification_decision", "").strip().lower()
                if classification_decision in ["accept classification", "accept_classification"]:
                    confidence_message = (
                        "Ticket is not duplicate. Moderate confidence match. Agent accepted classification. "
                        "Updating JIRA status to 'classified'."
                    )
                    default_status = "classified"
                elif classification_decision in ["reject classification", "reject_classification"]:
                    confidence_message = (
                        "Ticket is not duplicate. Moderate confidence match. Agent rejected classification. "
                        "Updating JIRA status to 'Manual_Classification_Required'."
                    )
                    default_status = "Manual_Classification_Required"
                else:
                    confidence_message = (
                        "Ticket is moderate confidence. Awaiting agent decision for classification. "
                        "Please provide 'classification_decision' as 'accept classification' or 'reject classification'."
                    )
                    return jsonify({
                        "message": confidence_message,
                        "duplicate_debug": duplicate_debug
                    })
            else:
                confidence_message = (
                    "Ticket is not duplicate. Low confidence match. Not making any change in JIRA status. "
                    "Pushing ticket for manual classification with status 'Manual_Classification_Required'."
                )
                default_status = "Manual_Classification_Required"
        else:
            best_score = 0.0
            confidence_message = "No similar tickets found; cannot compute confidence."
            default_status = "Manual_Classification_Required"
        # Update the classification result with the final status
        try:
            classification_dict = json.loads(classification)
        except Exception as e:
            classification_dict = {}
        classification_dict["Status"] = default_status
        classification = json.dumps(classification_dict)
        const_query = normalize_text(f"{summary} {description}")
        context_relevance = evaluate_context_relevance(const_query, retrieved_tickets, embeddings)
        answer_relevance = evaluate_answer_relevance(const_query, classification, retrieved_tickets, embeddings)
        return jsonify({
            "message": confidence_message,
            "classification": classification,
            "similar_tickets": retrieved_tickets,
            "context_relevance": context_relevance,
            "answer_relevance": answer_relevance,
            "duplicate_debug": duplicate_debug
        })
    except Exception as e:
        logging.error(f"Error in process_ticket: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
