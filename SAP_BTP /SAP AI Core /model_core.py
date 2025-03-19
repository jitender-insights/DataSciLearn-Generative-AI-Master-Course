from flask import Flask, request, jsonify, render_template
import logging
import traceback
import json
from config.logger_config import LoggerConfig
from analysis.anlaysis import *  # <-- Ensure your import paths are correct
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

# Initialize models and vector store
llm, embeddings = AIModels.initialize_models()
vectorstore = VectorStore.get_vectorstore(embeddings)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_ticket', methods=['POST'])
def process_ticket():
    """
    Unified flow with two steps:
      Step "validate": Check if the ticket info (summary, description, company code, component) is sufficient.
      Step "final": Perform duplicate detection, classification, and assignment.
    """
    try:
        data = request.json
        if not data:
            logging.error("No JSON data received in process_ticket")
            return jsonify({"error": "No JSON data received"}), 400

        # Determine step of processing: default is "final" to maintain backward compatibility.
        step = data.get("step", "final").strip().lower()

        # Extract required fields
        summary = data.get("Summary", data.get("summary", "")).strip()
        description = data.get("Description", data.get("description", "")).strip()
        component = data.get("Component", data.get("component", "")).strip()
        company_code = data.get("Company_Code", data.get("company_code", "")).strip()
        attachments = data.get("Attachments", [])
        
        logging.debug(
            f"Process Ticket - Received: Summary='{summary}', Description='{description}', "
            f"Company_Code='{company_code}', Component='{component}'"
        )

        # Step 1: Validate ticket information only
        if step == "validate":
            analysis_result = TicketAnalyzer.analyze_ticket_data(
                summary=summary,
                description=description,
                component=component,
                company_code=company_code,
                attachments=attachments
            )
            if analysis_result["missing_info"]:
                logging.warning("Ticket has missing/insufficient information.")
                return jsonify({
                    "status": analysis_result["status"],
                    "message": analysis_result["message"],
                    "missing_fields": analysis_result["missing_fields"]
                }), 400
            else:
                return jsonify({
                    "message": "Ticket information is sufficient. Proceed to AMS screen.",
                    "ticket_data": {
                        "summary": summary,
                        "description": description,
                        "component": component,
                        "company_code": company_code
                    }
                })

        # Step 2 (or legacy flow): Process the full ticket (duplicate detection, classification, assignment)
        # Extract additional AMS fields if provided
        urgency = data.get("urgency", "").strip()
        labels = data.get("labels", "").strip()
        jira_ticket_option = data.get("jira_ticket_option", "").strip()

        # 1) Check for Missing Information
        analysis_result = TicketAnalyzer.analyze_ticket_data(
            summary=summary,
            description=description,
            component=component,
            company_code=company_code,
            attachments=attachments
        )
        if analysis_result["missing_info"]:
            logging.warning("Ticket has missing/insufficient information.")
            return jsonify({
                "status": analysis_result["status"],
                "message": analysis_result["message"],
                "missing_fields": analysis_result["missing_fields"]
            }), 400

        # 2) Duplicate Detection
        duplicate_result = DuplicateDetector.check_duplicate_ticket({
            "Summary": summary,
            "Description": description,
            "Company_Code": company_code,
            "Component": component
        }, vectorstore, embeddings)
        is_duplicate = duplicate_result.get("is_duplicate", False)
        duplicate_category = duplicate_result.get("duplicate_category", "No duplicate found")
        original_ticket_id = duplicate_result.get("original_ticket_id", None)
        similarity = duplicate_result.get("similarity", 0.0)
        reasoning = duplicate_result.get("reasoning", "")
        duplicate_debug = {
            "is_duplicate": is_duplicate,
            "duplicate_category": duplicate_category,
            "original_ticket_id": original_ticket_id,
            "similarity": similarity,
            "reasoning": reasoning
        }

        # 3) Classification (Always)
        retrieved_tickets = TicketRetrieval.retrieve_and_rerank_tickets(
            vectorstore, summary, description, k=5
        )
        classification = TicketClassifier.generate_response_with_prompt(
            llm, retrieved_tickets, summary, description
        )
        const_query = Utils.normalize_text(f"{summary} {description}")
        context_relevance = Evaluator.evaluate_context_relevance(const_query, retrieved_tickets, embeddings)
        answer_relevance = Evaluator.evaluate_answer_relevance(const_query, classification, retrieved_tickets, embeddings)

        # 4) Confidence Threshold Logic
        confidence_message = ""
        if retrieved_tickets:
            best_ticket = retrieved_tickets[0]
            best_score = best_ticket.get("Combined_Score", 0.0)
            if best_score >= config.CLASSIFICATION_HIGH_CONF_THRESHOLD:
                confidence_message = "High confidence classification. Set JIRA ticket status to 'Classified' automatically."
            elif best_score >= config.CLASSIFICATION_MODERATE_CONF_THRESHOLD:
                confidence_message = "Moderate confidence classification. Set JIRA ticket status to 'Human_Evaluation_Required'."
            else:
                confidence_message = "Low confidence classification. Set JIRA ticket status to 'Manual_Classification_Required'."
        else:
            best_score = 0.0
            confidence_message = "No similar tickets found; cannot compute classification confidence."

        # 5) Perform Assignment using Incident_Type from classification JSON
        incident_type = "others"
        try:
            parsed = json.loads(classification)
            incident_type = parsed.get("Incident_Type", "others").strip().lower()
        except Exception as e:
            logging.warning(f"Could not parse classification JSON. Defaulting to 'others'. Error: {str(e)}")

        # Here, we assume that the division is provided by the classification or user.
        # For this example, we assume the division is the same as the incident_type.
        # Additional AMS fields (urgency, labels, jira_ticket_option) could be integrated into ticket_info if needed.
        ticket_info = {
            "ticket_type": incident_type,
            "summary": summary,
            "description": description,
            "division": incident_type,
            "urgency": urgency,
            "labels": labels,
            "jira_ticket_option": jira_ticket_option
        }
        assignment_group = TicketAssignment.assign_ticket(ticket_info, TicketAssignment.load_agents())

        # 6) Final Response
        final_response = {
            "message": "",
            "duplicate_debug": duplicate_debug,
            "classification": classification,
            "confidence_message": confidence_message,
            "similar_tickets": retrieved_tickets,
            "context_relevance": context_relevance,
            "answer_relevance": answer_relevance,
            "assignment_group": assignment_group
        }
        if is_duplicate:
            final_response["message"] = (
                "Ticket identified as DUPLICATE.\n"
                "Classification completed.\n"
            )
        else:
            final_response["message"] = "No duplicate found. Classification completed."
        
        logging.info(final_response["message"])
        return jsonify(final_response)

    except Exception as e:
        logging.error(f"Error in process_ticket: {str(e)}")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

# Additional Endpoint: Chart Data / Insights from Agent CSV
@app.route('/chart_data')
def chart_data():
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

# Additional Endpoint: Detailed Agent Data
@app.route('/agents_data')
def agents_data():
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
