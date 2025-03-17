@app.route('/process_ticket', methods=['POST'])
def process_ticket():
    """
    Updated to classify even if a ticket is a strong/combined duplicate.
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        summary = data.get("Summary", data.get("summary", "")).strip()
        description = data.get("Description", data.get("description", "")).strip()
        component = data.get("Component", data.get("component", "")).strip()
        company_code = data.get("Company_Code", data.get("company_code", "")).strip()

        duplicate_decision = data.get("duplicate_decision", "").strip().lower()
        classification_decision = data.get("classification_decision", "").strip().lower()

        if not summary or not description or not component or not company_code:
            return jsonify({"error": "Summary, Description, Company Code, and Component are mandatory."}), 400

        # Step 1: Check for duplicates
        duplicate_result = check_duplicate_ticket(
            {"Summary": summary, "Description": description, "Company_Code": company_code, "Component": component},
            vectorstore,
            embeddings
        )

        duplicate_category = duplicate_result.get("duplicate_category", "")
        similarity = duplicate_result.get("similarity", 0.0)
        is_duplicate_flag = duplicate_result.get("is_duplicate", False)
        original_ticket_id = duplicate_result.get("original_ticket_id", "")

        duplicate_debug = {
            "duplicate_category": duplicate_category,
            "is_duplicate": str(is_duplicate_flag),
            "original_ticket_id": str(original_ticket_id),
            "similarity": str(similarity),
            "reasoning": duplicate_result.get("reasoning", "")
        }

        # Step 2: If Strong Duplicate, create subtask but also classify
        if is_duplicate_flag and duplicate_category in ["Likely duplicate", "Combined duplicate"]:
            subtask = create_subtask(original_ticket_id, {"Description": description})

            # Perform classification even if it's a duplicate
            retrieved_tickets = retrieve_and_rerank_tickets(vectorstore, summary, description, k=5)
            classification = generate_response_with_prompt(llm, retrieved_tickets, summary, description)

            return jsonify({
                "message": "Duplicate ticket detected. Creating subtask and classifying the ticket.",
                "subtask": subtask,
                "classification": classification,
                "duplicate_debug": duplicate_debug
            })

        # Step 3: If "May Be Duplicate", check agent's decision
        elif duplicate_category.lower() == "may be duplicate":
            if duplicate_decision in ["accept duplicate", "accept_duplicate"]:
                subtask = create_subtask(original_ticket_id, {"Description": description})
                return jsonify({
                    "message": "Ticket flagged as 'May be duplicate' and agent accepted duplicate. Creating subtask.",
                    "subtask": subtask,
                    "duplicate_debug": duplicate_debug
                })
            elif duplicate_decision in ["reject duplicate", "reject_duplicate"]:
                pass  # Proceed with classification
            else:
                return jsonify({
                    "message": "Awaiting agent decision. Provide 'duplicate_decision' as 'accept duplicate' or 'reject duplicate'.",
                    "duplicate_debug": duplicate_debug
                })

        # Step 4: Classify the ticket (either after rejecting duplicate or if it's not a duplicate)
        retrieved_tickets = retrieve_and_rerank_tickets(vectorstore, summary, description, k=5)
        classification = generate_response_with_prompt(llm, retrieved_tickets, summary, description)

        best_score = retrieved_tickets[0]["Combined_Score"] if retrieved_tickets else 0.0
        default_status = "Manual_Classification_Required"

        if best_score >= config.CLASSIFICATION_HIGH_CONF_THRESHOLD:
            confidence_message = "High confidence classification. Marking as 'classified'."
            default_status = "classified"
        elif best_score >= config.CLASSIFICATION_MODERATE_CONF_THRESHOLD:
            if classification_decision in ["accept classification", "accept_classification"]:
                confidence_message = "Agent accepted classification. Marking as 'classified'."
                default_status = "classified"
            elif classification_decision in ["reject classification", "reject_classification"]:
                confidence_message = "Agent rejected classification. Marking as 'Manual_Classification_Required'."
                default_status = "Manual_Classification_Required"
            else:
                return jsonify({
                    "message": "Awaiting agent decision. Provide 'classification_decision' as 'accept classification' or 'reject classification'.",
                    "duplicate_debug": duplicate_debug
                })
        else:
            confidence_message = "Low confidence classification. Marking as 'Manual_Classification_Required'."

        # Update classification JSON with final status
        classification_dict = json.loads(classification)
        classification_dict["Status"] = default_status
        classification = json.dumps(classification_dict)

        return jsonify({
            "message": confidence_message,
            "classification": classification,
            "duplicate_debug": duplicate_debug
        })

    except Exception as e:
        logging.error(f"Error in process_ticket: {str(e)}")
        return jsonify({"error": str(e)}), 500
