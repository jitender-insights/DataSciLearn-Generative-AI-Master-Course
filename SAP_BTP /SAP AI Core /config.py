<!DOCTYPE html>
<html>
<head>
  <title>JIRA Ticket Classification & Duplicate Detection</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      max-width: 900px;
      margin: 0 auto;
      padding: 20px;
      background-color: #f8f9fa;
    }
    h1, h2, h3 {
      text-align: center;
      color: #333;
    }
    .form-group {
      margin-bottom: 15px;
    }
    label {
      font-weight: bold;
    }
    input[type="text"],
    textarea,
    select {
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 5px;
    }
    button {
      background-color: #007bff;
      color: white;
      padding: 12px 18px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      width: 100%;
      margin-top: 10px;
    }
    /* Loader Modal Styles */
    .modal {
      display: none;
      position: fixed;
      z-index: 999;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      overflow: auto;
      background-color: rgba(0,0,0,0.4);
    }
    .modal-content {
      background-color: #fefefe;
      margin: 15% auto;
      padding: 20px;
      border: 1px solid #888;
      width: 80%;
      max-width: 400px;
      text-align: center;
      border-radius: 5px;
    }
    .loader {
      border: 8px solid #f3f3f3;
      border-top: 8px solid #007bff;
      border-radius: 50%;
      width: 60px;
      height: 60px;
      animation: spin 1s linear infinite;
      margin: 0 auto 20px;
    }
    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
    /* Screen display */
    #screen2 {
      display: none;
    }
    /* Initially hide results, agent insights, and debug until classification is done */
    #resultsSection, #agentInsights, #debugSection {
      display: none;
      margin-top: 20px;
    }
    .debug-toggle {
      background-color: #28a745;
      color: white;
      padding: 10px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      width: 100%;
      margin-top: 10px;
    }
    /* Table & progress bar styles for debug info */
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    th, td {
      border: 1px solid #ddd;
      padding: 8px;
      text-align: left;
      font-size: 14px;
    }
    th {
      background-color: #007bff;
      color: #fff;
    }
    .progress-bar {
      height: 20px;
      background-color: #e0e0e0;
      border-radius: 10px;
      margin-top: 5px;
      overflow: hidden;
      width: 100px;
    }
    .progress {
      height: 100%;
      background-color: #4CAF50;
      text-align: center;
      line-height: 20px;
      color: white;
    }
  </style>
</head>
<body>
  <h1>JIRA Ticket Classification & Duplicate Detection</h1>
  
  <!-- Screen 1: Ticket Creation -->
  <div id="screen1">
    <div class="form-group">
      <label for="summary">Summary:</label>
      <input type="text" id="summary" name="summary" placeholder="Enter a brief summary">
    </div>
    <div class="form-group">
      <label for="description">Description:</label>
      <textarea id="description" name="description" placeholder="Enter a detailed description" rows="10"></textarea>
    </div>
    <div class="form-group">
      <label for="component">Component:</label>
      <input type="text" id="component" name="component">
    </div>
    <div class="form-group">
      <label for="company_code">Company Code:</label>
      <input type="text" id="company_code" name="company_code">
    </div>
    <button onclick="validateTicket()">Create Ticket</button>
  </div>
  
  <!-- Loader Modal -->
  <div id="loaderModal" class="modal">
    <div class="modal-content">
      <div class="loader"></div>
      <p>Ticket is sufficient for classification. Please wait...</p>
    </div>
  </div>
  
  <!-- Screen 2: AMS (Classification) -->
  <div id="screen2">
    <h2>AMS Ticket Screen</h2>
    <!-- Pre-filled fields (editable) -->
    <div class="form-group">
      <label for="summary2">Summary:</label>
      <input type="text" id="summary2" name="summary2">
    </div>
    <div class="form-group">
      <label for="description2">Description:</label>
      <textarea id="description2" name="description2" rows="10"></textarea>
    </div>
    <div class="form-group">
      <label for="component2">Component:</label>
      <input type="text" id="component2" name="component2">
    </div>
    <div class="form-group">
      <label for="company_code2">Company Code:</label>
      <input type="text" id="company_code2" name="company_code2">
    </div>
    
    <!-- Button to trigger classification (hidden after click) -->
    <button id="classifyBtn" onclick="classifyTicket()">Classify Ticket</button>
    
    <!-- Results Section (populated after classification) -->
    <div id="resultsSection">
      <!-- Classification Fields -->
      <div class="form-group" id="incidentTypeGroup" style="display:none;">
        <label for="incidentType">Incident Type:</label>
        <input type="text" id="incidentType">
      </div>
      <div class="form-group" id="category1Group" style="display:none;">
        <label for="category1">Category1:</label>
        <input type="text" id="category1">
      </div>
      <div class="form-group" id="category2Group" style="display:none;">
        <label for="category2">Category2:</label>
        <input type="text" id="category2">
      </div>
      <div class="form-group" id="category3Group" style="display:none;">
        <label for="category3">Category3:</label>
        <input type="text" id="category3">
      </div>
      <div class="form-group" id="urgencyFieldGroup" style="display:none;">
        <label for="urgencyField">Urgency:</label>
        <input type="text" id="urgencyField">
      </div>
      
      <!-- Duplicate Info: Only shown if duplicate is detected -->
      <div id="duplicateInfoSection" style="display:none; margin-top:20px;">
        <div class="form-group">
          <label for="isDuplicate">Is Duplicate:</label>
          <input type="text" id="isDuplicate">
        </div>
        <div class="form-group">
          <label for="duplicateCategory">Duplicate Category:</label>
          <input type="text" id="duplicateCategory">
        </div>
        <div class="form-group">
          <label for="originalTicketId">Original Ticket ID:</label>
          <input type="text" id="originalTicketId">
        </div>
        <div class="form-group">
          <label for="similarityScore">Similarity Score:</label>
          <input type="text" id="similarityScore">
        </div>
        <div class="form-group">
          <label for="duplicateReasoning">Reasoning:</label>
          <textarea id="duplicateReasoning" rows="4"></textarea>
        </div>
      </div>
      
      <!-- Assignment Group: Only shown if available -->
      <div id="assignmentGroupSection" style="display:none; margin-top:20px;">
        <div class="form-group">
          <label for="assignmentGroup">Assignment Group:</label>
          <input type="text" id="assignmentGroup">
        </div>
      </div>
    </div>
    
    <!-- Agent Insights (shown only after classification) -->
    <div id="agentInsights">
      <h2>Agent Insights</h2>
      <button onclick="fetchInsights()">Refresh Insights</button>
      <div id="insights_data"></div>
    </div>
    
    <!-- Debug Info Section (shown only after classification) -->
    <div id="debugSection">
      <button class="debug-toggle" onclick="toggleDebug()">Show Debug Info</button>
      <div id="debugContent" style="display:none;">
        <h3>Similar Tickets</h3>
        <div id="similar_tickets"></div>
        <h3>Context Relevance</h3>
        <div id="context_relevance"></div>
        <h3>Answer Relevance</h3>
        <div id="answer_relevance"></div>
      </div>
    </div>
  </div>
  
  <script>
    /****************************************
     *  SCREEN 1: Validate & Create Ticket  *
     ****************************************/
    function validateTicket() {
      const summary = document.getElementById('summary').value.trim();
      const description = document.getElementById('description').value.trim();
      const component = document.getElementById('component').value.trim();
      const company_code = document.getElementById('company_code').value.trim();
      
      if (!summary || !description || !component || !company_code) {
        alert('Summary, Description, Company Code, and Component are mandatory.');
        return;
      }
      
      // Send only the required fields for validation
      fetch('/process_ticket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ summary, description, component, company_code, step: "validate" })
      })
      .then(response => response.json())
      .then(data => {
        if (data.error || data.missing_fields) {
          alert(data.message || "Missing required ticket information.");
          return;
        }
        // Show loader modal upon successful validation
        document.getElementById('loaderModal').style.display = 'block';
        setTimeout(() => {
          document.getElementById('loaderModal').style.display = 'none';
          document.getElementById('screen1').style.display = 'none';
          document.getElementById('screen2').style.display = 'block';
          // Pre-fill Screen 2 fields with values from Screen 1
          document.getElementById('summary2').value = summary;
          document.getElementById('description2').value = description;
          document.getElementById('component2').value = component;
          document.getElementById('company_code2').value = company_code;
        }, 2000);
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error.message);
      });
    }
    
    /****************************************
     *  SCREEN 2: Classify Ticket & Display *
     ****************************************/
    function classifyTicket() {
      const summary = document.getElementById('summary2').value.trim();
      const description = document.getElementById('description2').value.trim();
      const component = document.getElementById('component2').value.trim();
      const company_code = document.getElementById('company_code2').value.trim();
      // Note: We do NOT send any priority field in this final step.
      
      // Hide the classify button once clicked
      document.getElementById('classifyBtn').style.display = 'none';
      
      fetch('/process_ticket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ summary, description, component, company_code, step: "final" })
      })
      .then(response => response.json())
      .then(data => {
        if (data.error) {
          alert(data.error);
          // Restore the classify button on error
          document.getElementById('classifyBtn').style.display = 'block';
          return;
        }
        alert(data.message || "Classification completed.");
        
        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // 1) Classification fields
        if (data.classification) {
          try {
            const classificationObj = JSON.parse(data.classification);
            // Use keys as returned by your classifier (with underscores)
            setFieldValueOrHide("incidentType", "incidentTypeGroup", classificationObj.Incident_Type);
            setFieldValueOrHide("category1", "category1Group", classificationObj.Category_1);
            setFieldValueOrHide("category2", "category2Group", classificationObj.Category_2);
            setFieldValueOrHide("category3", "category3Group", classificationObj.Category_3);
            setFieldValueOrHide("urgencyField", "urgencyFieldGroup", classificationObj.Urgency);
          } catch (e) {
            console.warn("Could not parse classification JSON.");
            setFieldValueOrHide("incidentType", "incidentTypeGroup", data.classification);
          }
        }
        
        // 2) Duplicate info: Only show if duplicate is detected
        if (data.duplicate_debug && data.duplicate_debug.is_duplicate) {
          document.getElementById('duplicateInfoSection').style.display = 'block';
          document.getElementById('isDuplicate').value       = data.duplicate_debug.is_duplicate || "";
          document.getElementById('duplicateCategory').value = data.duplicate_debug.duplicate_category || "";
          document.getElementById('originalTicketId').value  = data.duplicate_debug.original_ticket_id || "";
          document.getElementById('similarityScore').value   = data.duplicate_debug.similarity || "";
          document.getElementById('duplicateReasoning').value= data.duplicate_debug.reasoning || "";
        } else {
          document.getElementById('duplicateInfoSection').style.display = 'none';
        }
        
        // 3) Assignment Group: Only show if available
        if (data.assignment_group) {
          document.getElementById('assignmentGroupSection').style.display = 'block';
          document.getElementById('assignmentGroup').value = data.assignment_group;
        } else {
          document.getElementById('assignmentGroupSection').style.display = 'none';
        }
        
        // Show agent insights & debug sections after classification
        document.getElementById('agentInsights').style.display = 'block';
        document.getElementById('debugSection').style.display   = 'block';
        
        // Populate debug details if available
        if (data.similar_tickets) {
          buildSimilarTicketsTable(data.similar_tickets, 'similar_tickets');
        }
        if (data.context_relevance) {
          buildContextRelevanceTable(data.context_relevance, 'context_relevance');
        }
        if (data.answer_relevance) {
          buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
        }
      })
      .catch(error => {
        console.error('Error:', error);
        document.getElementById('classifyBtn').style.display = 'block';
        alert('Error: ' + error.message);
      });
    }
    
    // Helper: Set a field's value or hide its group if empty.
    function setFieldValueOrHide(fieldId, groupId, value) {
      if (value && value.trim() !== "") {
        document.getElementById(fieldId).value = value;
        document.getElementById(groupId).style.display = 'block';
      } else {
        document.getElementById(groupId).style.display = 'none';
      }
    }
    
    /****************************************
     *         Debug / Insights Utils        *
     ****************************************/
    function toggleDebug() {
      const debugContent = document.getElementById('debugContent');
      debugContent.style.display = (debugContent.style.display === 'none' || debugContent.style.display === '') ? 'block' : 'none';
    }
    
    function fetchInsights() {
      fetch('/chart_data')
      .then(response => response.json())
      .then(data => {
        let html = '<table><tr><th>Division</th><th>Total Agents</th><th>Avg Workload</th><th>Agents On Leave</th><th>Agents Online</th><th>Agents Available</th></tr>';
        for (const div in data) {
          html += `<tr>
            <td>${div}</td>
            <td>${data[div].total_agents}</td>
            <td>${data[div].avg_workload}</td>
            <td>${data[div].agents_on_leave}</td>
            <td>${data[div].agents_online}</td>
            <td>${data[div].agents_available}</td>
          </tr>`;
        }
        html += '</table>';
        document.getElementById('insights_data').innerHTML = html;
      })
      .catch(err => console.error("Error fetching insights:", err));
    }
    
    function buildSimilarTicketsTable(tickets, containerId) {
      if (!tickets || tickets.length === 0) {
        document.getElementById(containerId).innerHTML = 'No similar tickets found.';
        return;
      }
      let html = '<table><tr><th>Incident Type</th><th>Category1</th><th>Category2</th><th>Category3</th><th>Priority</th><th>Similarity</th><th>Cross-Encoder</th><th>Combined Score</th></tr>';
      tickets.forEach(ticket => {
        html += `<tr>
          <td>${ticket.Incident_Type}</td>
          <td>${ticket.Category1}</td>
          <td>${ticket.Category2}</td>
          <td>${ticket.Category3}</td>
          <td>${ticket.Priority}</td>
          <td>${ticket.Similarity_Score}</td>
          <td>${ticket.Cross_Encoder_Score}</td>
          <td>${ticket.Combined_Score}</td>
        </tr>`;
      });
      html += '</table>';
      document.getElementById(containerId).innerHTML = html;
    }
    
    function buildContextRelevanceTable(contextData, containerId) {
      if (!contextData) {
        document.getElementById(containerId).innerHTML = 'No context relevance data.';
        return;
      }
      let html = '<table><tr><th>Metric</th><th>Value</th><th>Visualization</th></tr>';
      for (const [key, val] of Object.entries(contextData)) {
        if (typeof val === 'number') {
          const pct = Math.round(val * 100);
          html += `<tr><td>${key}</td><td>${val}</td><td><div class="progress-bar"><div class="progress" style="width:${pct}%">${pct}%</div></div></td></tr>`;
        } else {
          html += `<tr><td>${key}</td><td colspan="2">${val}</td></tr>`;
        }
      }
      html += '</table>';
      document.getElementById(containerId).innerHTML = html;
    }
    
    function buildAnswerRelevanceTable(answerData, containerId) {
      if (!answerData) {
        document.getElementById(containerId).innerHTML = 'No answer relevance data.';
        return;
      }
      let html = '<table><tr><th>Metric</th><th>Value</th><th>Visualization</th></tr>';
      for (const [key, val] of Object.entries(answerData)) {
        if (typeof val === 'number') {
          const pct = Math.round(val * 100);
          html += `<tr><td>${key}</td><td>${val}</td><td><div class="progress-bar"><div class="progress" style="width:${pct}%">${pct}%</div></div></td></tr>`;
        } else {
          html += `<tr><td>${key}</td><td colspan="2">${val}</td></tr>`;
        }
      }
      html += '</table>';
      document.getElementById(containerId).innerHTML = html;
    }
  </script>
</body>
</html>
