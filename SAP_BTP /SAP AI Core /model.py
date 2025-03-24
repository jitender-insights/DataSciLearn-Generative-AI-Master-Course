import logging
import traceback
import json
import os
import csv
import random
import base64
from io import BytesIO
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, HTMLResponse

# ---------------------------
# Import your project modules
# ---------------------------
from config.logger_config import LoggerConfig
from analysis.anlaysis import TicketAnalyzer  # Ensure correct module name; note the typo if needed.
from classification.classifier import TicketClassifier
from duplicate.duplicate_detector import DuplicateDetector
from models.ai_models import AIModels
from models.vector_store import VectorStore
from models.ticket_retrieval import TicketRetrieval
from evaluation.evaluator import Evaluator
from utils.utils import Utils
from config import config
from assignement.assignement import TicketAssignment
from prompts.prompt_generator import PromptGenerator

# Database functions
from db.database import (
    insert_ticket_request,
    get_hana_connection,
    load_config,
    create_requests_table,
    create_classification_table,
)

# ---------------------------
# Set up Logger and App
# ---------------------------
LoggerConfig.setup_logger()
app = FastAPI(title="Ticket Processing API")

# Initialize models and vector store (shared by services)
llm, embeddings = AIModels.initialize_models()
vectorstore = VectorStore.get_vectorstore(embeddings)

# ---------------------------
# Home Endpoint
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = """
    <html>
      <head><title>Ticket Processing API</title></head>
      <body>
        <h1>Welcome to the Ticket Processing API</h1>
        <p>Use the provided endpoints to create and process tickets.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# ============================
# Database Endpoints
# ============================

@app.post("/create_ticket")
async def create_ticket_endpoint(request: Request):
    """
    Inserts a new ticket into the REQUESTS table.
    """
    try:
        data = await request.json()
        ticket_id = insert_ticket_request(data)
        return {"message": "Ticket created successfully", "ticket_id": ticket_id}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pending_tickets")
async def pending_tickets():
    """
    Retrieves tickets from the REQUESTS table with STATE = 'Request'
    """
    try:
        config_obj = load_config()
        schema = config_obj['schemas']['requests']
        conn = get_hana_connection(schema)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {schema}.REQUESTS WHERE STATE = 'Request'")
        columns = [col[0] for col in cursor.description]
        tickets = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return {"pending_tickets": tickets}
    except Exception as e:
        logging.error("Error fetching pending tickets: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/classified_tickets")
async def classified_tickets():
    """
    Retrieves all records from the CLASSIFICATIONS table.
    """
    try:
        config_obj = load_config()
        schema = config_obj['schemas']['classifications']
        conn = get_hana_connection(schema)
        cursor = conn.cursor()
        query = f"SELECT * FROM {schema}.TICKETS"
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        tickets = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return tickets
    except Exception as e:
        logging.error("Error fetching classified tickets: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Service Functions & Endpoints
# ============================

# --- Analysis Service ---
def perform_analysis(ticket_data: dict) -> dict:
    """
    Analyze ticket data for missing or insufficient information.
    """
    summary = ticket_data.get("Summary", ticket_data.get("summary", "")).strip()
    description = ticket_data.get("Description", ticket_data.get("description", "")).strip()
    component = ticket_data.get("Component", ticket_data.get("component", "")).strip()
    company_code = ticket_data.get("Company_Code", ticket_data.get("company_code", "")).strip()
    attachments = ticket_data.get("Attachments", [])
    
    logging.debug(f"Analysis Service: summary='{summary}', description='{description}', "
                  f"component='{component}', company_code='{company_code}'")
    
    analysis_result = TicketAnalyzer.analyze_ticket_data(
        summary=summary,
        description=description,
        component=component,
        company_code=company_code,
        attachments=attachments
    )
    return analysis_result

@app.post("/service/analysis")
async def analysis_endpoint(request: Request):
    """
    Analysis endpoint: validates ticket information.
    """
    try:
        data = await request.json()
        if not data:
            logging.error("No JSON data received in analysis_endpoint")
            raise HTTPException(status_code=400, detail="No JSON data received")
        result = perform_analysis(data)
        if result.get("missing_info", False):
            logging.warning("Ticket missing required information")
            return JSONResponse(
                status_code=400,
                content={
                    "status": result.get("status", "fail"),
                    "message": result.get("message", "Missing information"),
                    "missing_fields": result.get("missing_fields", [])
                }
            )
        return {"status": "success", "analysis": result}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# --- Duplicate Detection Service ---
def perform_duplicate(ticket_data: dict) -> dict:
    """
    Check for duplicate tickets.
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

@app.post("/service/duplicate")
async def duplicate_endpoint(request: Request):
    """
    Duplicate detection endpoint.
    """
    try:
        data = await request.json()
        if not data:
            logging.error("No JSON data received in duplicate_endpoint")
            raise HTTPException(status_code=400, detail="No JSON data received")
        result = perform_duplicate(data)
        return {"status": "success", "duplicate": result}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# --- Classification Service ---
def perform_classification(ticket_data: dict) -> dict:
    """
    Retrieve similar tickets and classify the ticket.
    """
    summary = ticket_data.get("Summary", ticket_data.get("summary", "")).strip()
    description = ticket_data.get("Description", ticket_data.get("description", "")).strip()

    retrieved_tickets = TicketRetrieval.retrieve_and_rerank_tickets(
        vectorstore, summary, description, k=5
    )
    classification = PromptGenerator.generate_response_with_prompt(
        llm, retrieved_tickets, summary, description
    )
    
    # Evaluate relevance/confidence
    const_query = Utils.normalize_text(f"{summary} {description}")
    context_relevance = Evaluator.evaluate_context_relevance(const_query, retrieved_tickets, embeddings)
    answer_relevance = Evaluator.evaluate_answer_relevance(const_query, classification, retrieved_tickets, embeddings)
    
    return {
        "classification": classification,
        "similar_tickets": retrieved_tickets,
        "context_relevance": context_relevance,
        "answer_relevance": answer_relevance
    }

@app.post("/service/classification")
async def classification_endpoint(request: Request):
    """
    Classification endpoint.
    """
    try:
        data = await request.json()
        if not data:
            logging.error("No JSON data received in classification_endpoint")
            raise HTTPException(status_code=400, detail="No JSON data received")
        result = perform_classification(data)
        return {"status": "success", "classification_result": result}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# --- Assignment Service ---
def perform_assignment(ticket_data: dict) -> dict:
    """
    Determine the assignment based on classification.
    """
    # Extract classification result if available; default incident type is 'others'
    incident_type = "others"
    classification = ticket_data.get("classification", None)
    if classification:
        try:
            parsed = json.loads(classification)
            incident_type = parsed.get("Incident_Type", "others").strip().lower()
        except Exception as e:
            logging.warning(f"Assignment: Error parsing classification JSON: {str(e)}")
    
    ticket_info = {
        "ticket_type": incident_type,
        "summary": ticket_data.get("Summary", ticket_data.get("summary", "")).strip(),
        "description": ticket_data.get("Description", ticket_data.get("description", "")).strip(),
        "division": incident_type
    }
    
    assignment_group = TicketAssignment.assign_ticket(ticket_info, TicketAssignment.load_agents())
    return {"assignment_group": assignment_group}

@app.post("/service/assignment")
async def assignment_endpoint(request: Request):
    """
    Assignment endpoint.
    """
    try:
        data = await request.json()
        if not data:
            logging.error("No JSON data received in assignment_endpoint")
            raise HTTPException(status_code=400, detail="No JSON data received")
        result = perform_assignment(data)
        return {"status": "success", "assignment_result": result}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# --- Master Orchestrator: Process Ticket ---
def update_or_insert_ticket_classification(ticket_id, classification, classification_confidence, assignment,
                                             duplicate_ticket_list, duplicate_similarity_score, new_state):
    """
    Updates (or inserts if missing) the classification record into the CLASSIFICATIONS table.
    """
    try:
        schema = load_config()['schemas']['classifications']
        conn = get_hana_connection(schema)
        cursor = conn.cursor()
        update_query = f"""
            UPDATE {schema}.TICKETS
            SET SUMMARY = (SELECT SUMMARY FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                DESCRIPTION = (SELECT DESCRIPTION FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                COMPONENT = (SELECT COMPONENT FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                COMPANY_CODE = (SELECT COMPANY_CODE FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                PRIORITY = (SELECT PRIORITY FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                CLASSIFICATION = ?,
                CLASSIFICATION_CONFIDENCE = ?,
                ASSIGNMENT_GROUP = ?,
                ASSIGNED_AGENT_ID = ?,
                ASSIGNED_AGENT_NAME = ?,
                ASSIGNMENT_CONFIDENCE = ?,
                JIRA_STATE = ?
            WHERE TICKET_ID = ?
        """
        params = (
            ticket_id, ticket_id, ticket_id, ticket_id, ticket_id,
            classification,
            round(classification_confidence * 100, 0),
            assignment.get("agent_group"),
            assignment.get("agent_id"),
            assignment.get("agent_name"),
            round(assignment.get("assignment_confidence") * 100, 0),
            new_state,
            ticket_id
        )
        cursor.execute(update_query, params)
        conn.commit()
        if cursor.rowcount == 0:
            insert_query = f"""
                INSERT INTO {schema}.TICKETS (
                    TICKET_ID, SUMMARY, DESCRIPTION, COMPONENT, COMPANY_CODE, PRIORITY,
                    CLASSIFICATION, CLASSIFICATION_CONFIDENCE, INCIDENT_TYPE,
                    CATEGORY_1, CATEGORY_2, CATEGORY_3, URGENCY, IMPACT,
                    DUPLICATE_TICKET_LIST, Duplicate_SIMILARITY_SCORE,
                    ASSIGNMENT_GROUP, ASSIGNED_AGENT_ID, ASSIGNED_AGENT_NAME, ASSIGNMENT_CONFIDENCE, JIRA_STATE
                )
                SELECT
                    r.TICKET_ID, r.SUMMARY, r.DESCRIPTION, r.COMPONENT, r.COMPANY_CODE, r.PRIORITY,
                    ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?, ?, ?,
                    ?
                FROM REQUESTS_DB.REQUESTS r
                WHERE r.TICKET_ID = ?
            """
            parsed = json.loads(classification)
            incident_type = parsed.get("Incident_Type", "unknown")
            category_1 = parsed.get("Category_1", "unknown")
            category_2 = parsed.get("Category_2", "unknown")
            category_3 = parsed.get("Category_3", "unknown")
            urgency = parsed.get("Urgency", "unknown")
            impact = "unknown"
            insert_params = (
                classification,
                round(classification_confidence * 100, 0),
                incident_type,
                category_1,
                category_2,
                category_3,
                urgency,
                impact,
                duplicate_ticket_list,
                duplicate_similarity_score,
                assignment.get("agent_group"),
                assignment.get("agent_id"),
                assignment.get("agent_name"),
                round(assignment.get("assignment_confidence") * 100, 0),
                new_state,
                ticket_id
            )
            cursor.execute(insert_query, insert_params)
            conn.commit()
            if cursor.rowcount == 0:
                error_msg = f"Error: Inserting classification record failed for ticket {ticket_id}."
                logging.error(error_msg)
                raise Exception(error_msg)
            else:
                logging.info(f"Classification record inserted for ticket {ticket_id}.")
        else:
            logging.info(f"Classification record updated for ticket {ticket_id}.")
        cursor.close()
        conn.close()
    except Exception as e:
        error_msg = f"Error updating/inserting classification for ticket {ticket_id}: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)


@app.post("/process_ticket")
async def process_ticket_endpoint(request: Request):
    """
    Master orchestrator endpoint.
    Processes a ticket through analysis, duplicate detection, classification, assignment,
    and then updates/inserts the classification record.
    Supports a 'restart_from' parameter in the JSON to restart from a failed step.
    """
    try:
        data = await request.json()
        if not data:
            logging.error("No JSON data received in process_ticket")
            raise HTTPException(status_code=400, detail="No JSON data received")
        restart_from = data.get("restart_from", "analysis")
        overall_result = {"ticket_data": data}
        
        # ----- Step 1: Analysis -----
        if restart_from == "analysis":
            analysis_result = perform_analysis(data)
            if analysis_result.get("missing_info", False):
                msg = "Missing or insufficient ticket information."
                logging.warning(msg)
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": msg,
                        "service": "analysis",
                        "analysis": analysis_result
                    }
                )
            overall_result["analysis"] = analysis_result
            restart_from = "duplicate"
        
        # ----- Step 2: Duplicate Detection -----
        if restart_from == "duplicate":
            duplicate_result = perform_duplicate(data)
            overall_result["duplicate"] = duplicate_result
            restart_from = "classification"
        
        # ----- Step 3: Classification -----
        if restart_from == "classification":
            classification_result = perform_classification(data)
            overall_result["classification_result"] = classification_result
            # Determine confidence from best similar ticket
            retrieved_tickets = classification_result.get("similar_tickets", [])
            if retrieved_tickets:
                best_ticket = retrieved_tickets[0]
                classification_confidence = best_ticket.get("Combined_Score", 0.0)
            else:
                classification_confidence = 0.0
            overall_result["classification_confidence"] = classification_confidence
            # Validate classification JSON output
            try:
                parsed = json.loads(classification_result["classification"])
                incident_type = parsed.get("Incident_Type", "").strip().lower()
                if not incident_type:
                    raise Exception("Classification module returned empty Incident_Type")
            except Exception as e:
                logging.error(f"Error parsing classification JSON: {str(e)}")
                return JSONResponse(
                    status_code=500,
                    content={"error": "Classification modules failed"}
                )
            restart_from = "assignment"
        
        # ----- Step 4: Assignment -----
        if restart_from == "assignment":
            assignment = TicketAssignment.assign_ticket({
                "ticket_type": incident_type,
                "summary": data.get("Summary", data.get("summary", "")).strip(),
                "description": data.get("Description", data.get("description", "")).strip(),
                "division": incident_type
            }, TicketAssignment.load_agents())
            overall_result["assignment"] = assignment
            
        # ----- Update/Insert Classification Record -----
        ticket_id = data.get("ticket_id")
        if ticket_id:
            update_or_insert_ticket_classification(
                ticket_id,
                classification_result["classification"],
                classification_confidence,
                assignment,
                duplicate_result.get("original_ticket_id"),
                duplicate_result.get("similarity", 0.0),
                new_state="open"
            )
        else:
            logging.error("No ticket_id provided for classification update.")
            return JSONResponse(status_code=400, content={"error": "No ticket_id provided"})
        
        final_response = {
            "message": "Classification completed.",
            "duplicate_debug": {
                "is_duplicate": duplicate_result.get("is_duplicate", False),
                "duplicate_category": duplicate_result.get("duplicate_category", "No duplicate found"),
                "original_ticket_id": duplicate_result.get("original_ticket_id"),
                "similarity": duplicate_result.get("similarity", 0.0),
                "reasoning": duplicate_result.get("reasoning", "")
            },
            "classification": classification_result["classification"],
            "classification_confidence": classification_confidence,
            "confidence_message": (
                "High confidence classification." if classification_confidence >= config.CLASSIFICATION_HIGH_CONF_THRESHOLD
                else "Moderate confidence classification." if classification_confidence >= config.CLASSIFICATION_MODERATE_CONF_THRESHOLD
                else "Low confidence classification."
            ),
            "similar_tickets": classification_result["similar_tickets"],
            "assignment_group": assignment.get("agent_group"),
            "agent_id": assignment.get("agent_id"),
            "agent_name": assignment.get("agent_name"),
            "assignment_confidence": assignment.get("assignment_confidence")
        }
        if duplicate_result.get("is_duplicate", False):
            final_response["message"] = "Ticket identified as DUPLICATE. Classification completed."
        logging.info(final_response["message"])
        return final_response
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Chart Data Endpoint
# ============================
@app.get("/chart_data")
async def chart_data():
    """
    Returns agent statistics based on current load and past SLA breaches.
    Also returns graphs (as base64-encoded PNGs) for visualization.
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
            stats[div]["total_agents"] += 1
            stats[div]["avg_workload"] += a["current_load"]
            if a["leave_flag"]:
                stats[div]["agents_on_leave"] += 1
            if a["Availability"]:
                stats[div]["agents_online"] += 1
                if not a["leave_flag"]:
                    stats[div]["agents_available"] += 1
        for div in stats:
            total = stats[div]["total_agents"]
            stats[div]["avg_workload"] = round(stats[div]["avg_workload"] / total, 2) if total > 0 else 0.0

        # Graph: Current Load
        agent_names = [a["agent_name"] for a in agents]
        current_loads = [a["current_load"] for a in agents]
        sla_breaches = [a.get("past_sla_breaches", 0) for a in agents]
        index = list(range(len(agents)))

        # Current Load graph
        fig1, ax1 = plt.subplots(figsize=(10, max(6, len(agents)*0.2)))
        ax1.barh(index, current_loads, label='Current Load')
        ax1.set_yticks(index)
        ax1.set_yticklabels(agent_names)
        ax1.invert_yaxis()
        ax1.set_xlabel('Current Load (Open Tickets)')
        ax1.set_title('Agent Current Load')
        ax1.legend()
        buf1 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf1, format='png')
        buf1.seek(0)
        graph_current_load = base64.b64encode(buf1.getvalue()).decode('utf-8')
        plt.close(fig1)

        # SLA Breaches graph
        fig2, ax2 = plt.subplots(figsize=(10, max(6, len(agents)*0.2)))
        ax2.barh(index, sla_breaches, label='Past SLA Breaches')
        ax2.set_yticks(index)
        ax2.set_yticklabels(agent_names)
        ax2.invert_yaxis()
        ax2.set_xlabel('Past SLA Breaches')
        ax2.set_title('Agent Past SLA Breaches')
        ax2.legend()
        buf2 = BytesIO()
        plt.tight_layout()
        plt.savefig(buf2, format='png')
        buf2.seek(0)
        graph_sla = base64.b64encode(buf2.getvalue()).decode('utf-8')
        plt.close(fig2)

        return {"agents": agents, "stats": stats, "graph_current_load": graph_current_load, "graph_sla": graph_sla}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================
# Agents Data Endpoint
# ============================
@app.get("/agents_data")
async def agents_data():
    """
    Returns the full list of agents.
    """
    try:
        agents = TicketAssignment.load_agents()
        return agents
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# Application Startup: Create Tables (if needed)
# ============================
@app.on_event("startup")
async def startup_event():
    try:
        create_requests_table()
        create_classification_table()
    except Exception as e:
        logging.error(traceback.format_exc())
        # Optionally, re-raise to abort startup
        raise e

# ---------------------------
# To run this app:
# Use the command: uvicorn main:app --host 0.0.0.0 --port 5000 --reload
# ---------------------------
