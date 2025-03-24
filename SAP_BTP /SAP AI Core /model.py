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
# Import project modules
# ---------------------------
from config.logger_config import LoggerConfig
from analysis.anlaysis import *   # Ensure this import does not pollute the namespace
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
from mappers.mappers import TicketRequest, ProcessTicketRequest

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
app = FastAPI(
    title="Ticket Processing API",
    description="A production-grade API for processing tickets that supports analysis, duplicate detection, classification, assignment, and reporting endpoints.",
    version="1.0.0"
)

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
@app.post("/create_ticket", summary="Create a New Ticket", description="Inserts a new ticket into the REQUESTS table.")
async def create_ticket_endpoint(ticket: TicketRequest):
    """
    Creates a new ticket based on the provided JSON body.
    Example JSON:
    {
      "Summary": "Network outage in building 5",
      "Description": "Multiple users report Wi-Fi connectivity loss.",
      "Component": "Network",
      "Company_Code": "XYZ123",
      "Attachments": []
    }
    """
    try:
        ticket_data = ticket.dict()
        ticket_id = insert_ticket_request(ticket_data)
        logging.info(f"Ticket created successfully with ID: {ticket_id}")
        return {
            "message": "Ticket created successfully",
            "ticket_id": ticket_id,
            "received_data": ticket_data
        }
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred while creating the ticket.")

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

@app.post("/service/analysis", summary="Ticket Analysis", description="Analyzes ticket information for missing or insufficient details.")
async def analysis_endpoint(ticket: TicketRequest):
    """
    Analyzes ticket information to check for missing or insufficient details.
    Example JSON:
    {
      "Summary": "Network outage in building 5",
      "Description": "Multiple users report Wi-Fi connectivity loss.",
      "Component": "Network",
      "Company_Code": "XYZ123",
      "Attachments": []
    }
    """
    try:
        data = ticket.dict()
        analysis_result = TicketAnalyzer.analyze_ticket_data(
            summary=data["Summary"],
            description=data["Description"],
            component=data["Component"],
            company_code=data["Company_Code"],
            attachments=data.get("Attachments", [])
        )
        if analysis_result.get("missing_info", False):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Missing or insufficient ticket information.",
                    "service": "analysis",
                    "analysis": analysis_result
                }
            )
        return {"status": "success", "analysis": analysis_result}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred during analysis.")

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

@app.post("/service/duplicate", summary="Duplicate Detection", description="Checks for duplicate tickets.")
async def duplicate_endpoint(ticket: TicketRequest):
    try:
        ticket_data = ticket.dict()
        result = DuplicateDetector.check_duplicate_ticket(ticket_data, vectorstore, embeddings)
        return {"status": "success", "duplicate": result}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred during duplicate detection.")

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
    
    const_query = Utils.normalize_text(f"{summary} {description}")
    context_relevance = Evaluator.evaluate_context_relevance(const_query, retrieved_tickets, embeddings)
    answer_relevance = Evaluator.evaluate_answer_relevance(const_query, classification, retrieved_tickets, embeddings)
    
    return {
        "classification": classification,
        "similar_tickets": retrieved_tickets,
        "context_relevance": context_relevance,
        "answer_relevance": answer_relevance
    }

@app.post("/service/classification", summary="Classification", description="Generates ticket classification based on similar tickets.")
async def classification_endpoint(ticket: TicketRequest):
    try:
        data = ticket.dict()
        retrieved_tickets = TicketRetrieval.retrieve_and_rerank_tickets(
            vectorstore, data["Summary"], data["Description"], k=5
        )
        classification = PromptGenerator.generate_response_with_prompt(
            llm, retrieved_tickets, data["Summary"], data["Description"]
        )
        const_query = Utils.normalize_text(f"{data['Summary']} {data['Description']}")
        context_relevance = Evaluator.evaluate_context_relevance(const_query, retrieved_tickets, embeddings)
        answer_relevance = Evaluator.evaluate_answer_relevance(const_query, classification, retrieved_tickets, embeddings)
        return {
            "status": "success",
            "classification_result": {
                "classification": classification,
                "similar_tickets": retrieved_tickets,
                "context_relevance": context_relevance,
                "answer_relevance": answer_relevance
            }
        }
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred during classification.")

# --- Assignment Service ---
def perform_assignment(ticket_data: dict) -> dict:
    """
    Determine the assignment based on classification.
    """
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

@app.post("/service/assignment", summary="Assignment", description="Assigns a ticket based on its classification.")
async def assignment_endpoint(ticket: TicketRequest):
    try:
        data = ticket.dict()
        classification = data.get("Classification", '{"Incident_Type": "others"}')
        try:
            parsed = json.loads(classification)
            incident_type = parsed.get("Incident_Type", "others").strip().lower()
        except Exception:
            incident_type = "others"
        ticket_info = {
            "ticket_type": incident_type,
            "summary": data["Summary"],
            "description": data["Description"],
            "division": incident_type
        }
        assignment = TicketAssignment.assign_ticket(ticket_info, TicketAssignment.load_agents())
        return {"status": "success", "assignment_result": assignment}
    except Exception as e:
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An error occurred during assignment.")

# --- Master Orchestrator: Update/Insert Classification ---
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
            SET SUMMARY = (SELECT r.SUMMARY FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                DESCRIPTION = (SELECT r.DESCRIPTION FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                COMPONENT = (SELECT r.COMPONENT FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                COMPANY_CODE = (SELECT r.COMPANY_CODE FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
                PRIORITY = (SELECT r.PRIORITY FROM REQUESTS_DB.REQUESTS WHERE TICKET_ID = ?),
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
            round((assignment.get("assignment_confidence") or 0) * 100, 0),
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
                round((assignment.get("assignment_confidence") or 0) * 100, 0),
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
        
        # Update the corresponding REQUESTS record to set STATE = 'open'
        req_schema = load_config()['schemas']['requests']
        req_conn = get_hana_connection(req_schema)
        req_cursor = req_conn.cursor()
        req_update = f"UPDATE {req_schema}.REQUESTS SET STATE = 'open' WHERE TICKET_ID = ?"
        req_cursor.execute(req_update, (ticket_id,))
        req_conn.commit()
        req_cursor.close()
        req_conn.close()
        logging.info(f"REQUESTS table updated: Ticket {ticket_id} state set to 'open'.")
    except Exception as e:
        error_msg = f"Error updating/inserting classification for ticket {ticket_id}: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

# --- Master Orchestrator: Process Ticket Endpoint ---
@app.post("/process_ticket", summary="Process Ticket", description="Processes a ticket through analysis, duplicate detection, classification, assignment, and then updates/inserts the classification record.")
async def process_ticket_endpoint(ticket: ProcessTicketRequest):
    try:
        data = ticket.dict()
        overall_result = {"ticket_data": data}
        incident_type = None  # Declare incident_type in outer scope

        # Step 1: Analysis
        if data.get("restart_from") == "analysis":
            analysis_result = perform_analysis(data)
            if analysis_result.get("missing_info", False):
                msg = "Missing or insufficient ticket information."
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": msg,
                        "service": "analysis",
                        "analysis": analysis_result
                    }
                )
            overall_result["analysis"] = analysis_result
            data["restart_from"] = "duplicate"
        
        # Step 2: Duplicate Detection
        if data.get("restart_from") == "duplicate":
            duplicate_result = perform_duplicate(data)
            overall_result["duplicate"] = duplicate_result
            data["restart_from"] = "classification"
        
        # Step 3: Classification
        if data.get("restart_from") == "classification":
            classification_result = perform_classification(data)
            overall_result["classification_result"] = classification_result
            retrieved_tickets = classification_result.get("similar_tickets", [])
            classification_confidence = (retrieved_tickets[0].get("Combined_Score", 0.0) if retrieved_tickets else 0.0)
            overall_result["classification_confidence"] = classification_confidence
            try:
                parsed = json.loads(classification_result["classification"])
                incident_type = parsed.get("Incident_Type", "").strip().lower()
                if not incident_type:
                    raise Exception("Classification module returned empty Incident_Type")
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Classification modules failed"}
                )
            data["restart_from"] = "assignment"
        
        # Step 4: Assignment
        if data.get("restart_from") == "assignment":
            if incident_type is None:
                incident_type = "others"
            assignment = TicketAssignment.assign_ticket({
                "ticket_type": incident_type,
                "summary": data.get("Summary", "").strip(),
                "description": data.get("Description", "").strip(),
                "division": incident_type
            }, TicketAssignment.load_agents())
            overall_result["assignment"] = assignment
        
        # Step 5: Update/Insert Classification Record
        ticket_id = data.get("ticket_id")
        if ticket_id:
            update_or_insert_ticket_classification(
                ticket_id,
                classification_result["classification"],
                overall_result["classification_confidence"],
                assignment,
                overall_result["duplicate"].get("original_ticket_id"),
                overall_result["duplicate"].get("similarity", 0.0),
                new_state="open"
            )
        else:
            return JSONResponse(status_code=400, content={"error": "No ticket_id provided"})
        
        overall_result["message"] = "Ticket processing completed successfully."
        return overall_result
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
        
        # Graph: Past SLA Breaches
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
        raise e
