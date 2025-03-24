from flask import Flask, request, jsonify, render_template
import logging
import traceback
import json

from config.logger_config import LoggerConfig
from analysis.analysis import TicketAnalyzer  # Ensure module name is correct
from classification.classifier import TicketClassifier
from duplicate.duplicate_detector import DuplicateDetector
from models.ai_models import AIModels
from models.vector_store import VectorStore
from models.ticket_retrieval import TicketRetrieval
from evaluation.evaluator import Evaluator
from utils.utils import Utils
from config import config
from assignement.assignement import TicketAssignment

# Set up logger
LoggerConfig.setup_logger()

app = Flask(__name__)

# Initialize models and vector store (these are reused across services)
llm, embeddings = AIModels.initialize_models()
vectorstore = VectorStore.get_vectorstore(embeddings)


# ------------------------------
# Analysis Service and Endpoint
# ------------------------------
def perform_analysis(ticket_data: dict) -> dict:
    """
    Performs analysis on ticket data.
    Returns the analysis_result dictionary.
    """
    summary = ticket_data.get("Summary", ticket_data.get("summary", "")).strip()
    description = ticket_data.get("Description", ticket_data.get("description", "")).strip()
    component = ticket_data.get("Component", ticket_data.get("component", "")).strip()
    company_code = ticket_data.get("Company_Code", ticket_data.get("company_code", "")).strip()
    attachments = ticket_data.get("Attachments", [])

    logging.debug(
        f"Analysis Service: Received summary='{summary}', description='{description}', "
        f"component='{component}', company_code='{company_code}'"
    )

    analysis_result = TicketAnalyzer.analyze_ticket_data(
        summary=summary,
        description=description,
        component=component,
        company_code=company_code,
        attachments=attachments
    )
    return analysis_result


@app.route('/service/analysis', methods=['POST'])
def analysis_endpoint():
    """
    Endpoint for analysis service.
    Expects ticket JSON and returns the analysis result.
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in analysis_endpoint")
            return jsonify({"error": "No JSON data received"}), 400

        analysis_result = perform_analysis(data)
        if analysis_result.get("missing_info", False):
            logging.warning("Analysis Service: Ticket has missing or insufficient information.")
            return jsonify({
                "status": analysis_result.get("status", "fail"),
                "message": analysis_result.get("message", "Missing information"),
                "missing_fields": analysis_result.get("missing_fields", [])
            }), 400

        return jsonify({"status": "success", "analysis": analysis_result})
    except Exception as e:
        logging.error(f"Error in analysis_endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# Duplicate Detection Service/Endpoint
# -----------------------------------
def perform_duplicate(ticket_data: dict) -> dict:
    """
    Performs duplicate detection using ticket data.
    Returns the duplicate detection result.
    """
    summary = ticket_data.get("Summary", ticket_data.get("summary", "")).strip()
    description = ticket_data.get("Description", ticket_data.get("description", "")).strip()
    component = ticket_data.get("Component", ticket_data.get("component", "")).strip()
    company_code = ticket_data.get("Company_Code", ticket_data.get("company_code", "")).strip()

    duplicate_result = DuplicateDetector.check_duplicate_ticket(
        {
            "Summary": summary,
            "Description": description,
            "Company_Code": company_code,
            "Component": component
        },
        vectorstore,
        embeddings
    )
    return duplicate_result


@app.route('/service/duplicate', methods=['POST'])
def duplicate_endpoint():
    """
    Endpoint for duplicate detection service.
    Expects ticket JSON and returns the duplicate detection result.
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in duplicate_endpoint")
            return jsonify({"error": "No JSON data received"}), 400

        duplicate_result = perform_duplicate(data)
        return jsonify({"status": "success", "duplicate": duplicate_result})
    except Exception as e:
        logging.error(f"Error in duplicate_endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Classification Service/Endpoint
# ------------------------------
def perform_classification(ticket_data: dict) -> dict:
    """
    Performs classification using ticket data.
    Returns the classification result dictionary.
    """
    summary = ticket_data.get("Summary", ticket_data.get("summary", "")).strip()
    description = ticket_data.get("Description", ticket_data.get("description", "")).strip()

    # Retrieve similar tickets first
    retrieved_tickets = TicketRetrieval.retrieve_and_rerank_tickets(
        vectorstore, summary, description, k=5
    )
    classification = TicketClassifier.generate_response_with_prompt(
        llm, retrieved_tickets, summary, description
    )

    # Evaluate context and answer relevance
    const_query = Utils.normalize_text(f"{summary} {description}")
    context_relevance = Evaluator.evaluate_context_relevance(const_query, retrieved_tickets, embeddings)
    answer_relevance = Evaluator.evaluate_answer_relevance(const_query, classification, retrieved_tickets, embeddings)

    result = {
        "classification": classification,
        "similar_tickets": retrieved_tickets,
        "context_relevance": context_relevance,
        "answer_relevance": answer_relevance
    }
    return result


@app.route('/service/classification', methods=['POST'])
def classification_endpoint():
    """
    Endpoint for classification service.
    Expects ticket JSON and returns the classification result.
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in classification_endpoint")
            return jsonify({"error": "No JSON data received"}), 400

        classification_result = perform_classification(data)
        return jsonify({"status": "success", "classification_result": classification_result})
    except Exception as e:
        logging.error(f"Error in classification_endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Assignment Service/Endpoint
# ------------------------------
def perform_assignment(ticket_data: dict) -> dict:
    """
    Performs ticket assignment based on the classification result.
    Expects ticket_data to include necessary information (e.g. classification result).
    """
    # Extract incident type from the classification result (if available), default to "others"
    incident_type = "others"
    classification = ticket_data.get("classification", None)
    if classification:
        try:
            parsed = json.loads(classification)
            incident_type = parsed.get("Incident_Type", "others").strip().lower()
        except Exception as e:
            logging.warning(
                f"Assignment Service: Failed to parse classification JSON. Defaulting to 'others'. Error: {str(e)}"
            )

    # Build the ticket information structure for assignment.
    ticket_info = {
        "ticket_type": incident_type,
        "summary": ticket_data.get("Summary", ticket_data.get("summary", "")).strip(),
        "description": ticket_data.get("Description", ticket_data.get("description", "")).strip(),
        "division": incident_type  # You might override this with user input if available.
    }

    # Invoke the assignment logic (using agent data loaded from CSV)
    assignment_group = TicketAssignment.assign_ticket(ticket_info, TicketAssignment.load_agents())
    return {"assignment_group": assignment_group}


@app.route('/service/assignment', methods=['POST'])
def assignment_endpoint():
    """
    Endpoint for the assignment service.
    Expects ticket JSON (which may include a classification result) and returns the assignment result.
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in assignment_endpoint")
            return jsonify({"error": "No JSON data received"}), 400

        assignment_result = perform_assignment(data)
        return jsonify({"status": "success", "assignment_result": assignment_result})
    except Exception as e:
        logging.error(f"Error in assignment_endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Master Orchestrator Endpoint
# ------------------------------
@app.route('/process_ticket', methods=['POST'])
def process_ticket_sequential():
    """
    Master orchestrator endpoint.
    Runs the services sequentially: analysis -> duplicate -> classification -> assignment.
    An optional parameter 'restart_from' can be passed in the request JSON
    to start processing from a specific service (e.g. analysis, duplicate, classification, assignment).
    If any service fails, it returns an error with details so that the client can retry.
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in process_ticket")
            return jsonify({"error": "No JSON data received"}), 400

        restart_from = data.get("restart_from", "analysis")
        overall_result = {"ticket_data": data}

        # Step 1: Analysis
        if restart_from == "analysis":
            try:
                analysis_result = perform_analysis(data)
                if analysis_result.get("missing_info", False):
                    error_msg = "Missing or insufficient ticket information."
                    logging.warning(f"Analysis failed: {error_msg}")
                    return jsonify({
                        "error": error_msg,
                        "service": "analysis",
                        "analysis": analysis_result
                    }), 400
                overall_result["analysis"] = analysis_result
                restart_from = "duplicate"
            except Exception as e:
                logging.error(f"Error in analysis step: {str(e)}")
                logging.error(traceback.format_exc())
                return jsonify({
                    "error": str(e),
                    "service": "analysis",
                    "ticket_data": data
                }), 500

        # Step 2: Duplicate Detection
        if restart_from == "duplicate":
            try:
                duplicate_result = perform_duplicate(data)
                overall_result["duplicate"] = duplicate_result
                restart_from = "classification"
            except Exception as e:
                logging.error(f"Error in duplicate step: {str(e)}")
                logging.error(traceback.format_exc())
                return jsonify({
                    "error": str(e),
                    "service": "duplicate",
                    "ticket_data": data,
                    "analysis": overall_result.get("analysis")
                }), 500

        # Step 3: Classification
        if restart_from == "classification":
            try:
                classification_result = perform_classification(data)
                overall_result["classification_result"] = classification_result

                # Example confidence logic based on best similar ticket score
                retrieved_tickets = classification_result.get("similar_tickets", [])
                if retrieved_tickets:
                    best_ticket = retrieved_tickets[0]
                    best_score = best_ticket.get("Combined_Score", 0.0)
                    if best_score >= config.CLASSIFICATION_HIGH_CONF_THRESHOLD:
                        overall_result["confidence_message"] = "High confidence classification. Set JIRA ticket status to 'Classified' automatically."
                    elif best_score >= config.CLASSIFICATION_MODERATE_CONF_THRESHOLD:
                        overall_result["confidence_message"] = "Moderate confidence classification. Set JIRA ticket status to 'Human_Evaluation_Required'."
                    else:
                        overall_result["confidence_message"] = "Low confidence classification. Set JIRA ticket status to 'Manual_Classification_Required'."
                else:
                    overall_result["confidence_message"] = "No similar tickets found; cannot compute classification confidence."
                restart_from = "assignment"
            except Exception as e:
                logging.error(f"Error in classification step: {str(e)}")
                logging.error(traceback.format_exc())
                return jsonify({
                    "error": str(e),
                    "service": "classification",
                    "ticket_data": data,
                    "analysis": overall_result.get("analysis"),
                    "duplicate": overall_result.get("duplicate")
                }), 500

        # Step 4: Assignment
        if restart_from == "assignment":
            try:
                assignment_result = perform_assignment(data)
                overall_result["assignment_result"] = assignment_result
            except Exception as e:
                logging.error(f"Error in assignment step: {str(e)}")
                logging.error(traceback.format_exc())
                return jsonify({
                    "error": str(e),
                    "service": "assignment",
                    "ticket_data": data,
                    "analysis": overall_result.get("analysis"),
                    "duplicate": overall_result.get("duplicate"),
                    "classification_result": overall_result.get("classification_result")
                }), 500

        overall_result["message"] = "Ticket processing completed successfully."
        logging.info(overall_result["message"])
        return jsonify(overall_result)

    except Exception as e:
        logging.error(f"Error in process_ticket_sequential: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ------------------------------
# Additional Endpoints
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chart_data')
def chart_data():
    """
    Reads the agent CSV (via TicketAssignment.load_agents()) and computes statistics.
    """
    try:
        agents = TicketAssignment.load_agents()
        stats = {}
        for a in agents:
            div = a["division"]
            if div not in stats:
                stats[div] = {
                    "total_agents": 0,
                    "avg_workload": 0.0,
                    "agents_on_leave": 0,
                    "agents_online": 0,
                    "agents_available": 0
                }
        for a in agents:
            div = a["division"]
            stats[div]["total_agents"] += 1
            stats[div]["avg_workload"] += a["current_load"]
            if a["leave_flag"]:
                stats[div]["agents_on_leave"] += 1
            if a["Availability"] and TicketAssignment.is_within_working_hours(a):
                stats[div]["agents_online"] += 1
                if not a["leave_flag"]:
                    stats[div]["agents_available"] += 1
        for div in stats:
            total = stats[div]["total_agents"]
            stats[div]["avg_workload"] = round(stats[div]["avg_workload"] / total, 2) if total > 0 else 0.0
        return jsonify(stats)
    except Exception as e:
        logging.error(f"Error in chart_data: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/agents_data')
def agents_data():
    """
    Returns the full list of agents from the CSV.
    """
    try:
        agents = TicketAssignment.load_agents()
        return jsonify(agents)
    except Exception as e:
        logging.error(f"Error in agents_data: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logging.error(f"Error starting application: {e}")
        logging.error(traceback.format_exc())
