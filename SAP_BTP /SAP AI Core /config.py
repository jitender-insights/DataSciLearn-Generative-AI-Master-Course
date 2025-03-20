<!DOCTYPE html>
<html>
<head>
  <title>JIRA Ticket Classification & Duplicate Detection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
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
    /* Screen 1 Header */
    #screen1Header {
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
    input[type="file"],
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
    /* Screen 2 (AMS View) – remains hidden in the original window */
    #screen2 {
      display: none;
      position: fixed;
      z-index: 1000;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 20px;
      background-color: #f8f9fa;
      overflow-y: auto;
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
    /* Collapsible toggles for debug & assignment insights */
    .toggle-btn {
      background-color: #6c757d;
      color: white;
      padding: 10px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      width: 100%;
      margin-top: 10px;
      text-align: left;
    }
    .toggle-content {
      display: none;
      margin-top: 10px;
    }
  </style>
</head>
<body>
  <!-- Screen 1: Ticket Creation -->
  <div id="screen1">
    <h1 id="screen1Header">User ticket creation view</h1>
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
      <label for="company_code">Company:</label>
      <input type="text" id="company_code" name="company_code">
    </div>
    <div class="form-group">
      <label for="priority">User Priority:</label>
      <select id="priority" name="priority">
        <option value="emergency">Emergency</option>
        <option value="high">High</option>
        <option value="medium" selected>Medium</option>
        <option value="low">Low</option>
      </select>
    </div>
    <div class="form-group">
      <label for="attachments">Attachments:</label>
      <input type="file" id="attachments" name="attachments" multiple>
    </div>
    <button onclick="validateTicket()">Create Ticket</button>
  </div>

  <!-- Loader Modal -->
  <div id="loaderModal" class="modal">
    <div class="modal-content">
      <div class="loader"></div>
      <p>Information sufficient for further processing, creating ticket....</p>
    </div>
  </div>

  <!-- Screen 2: AMS (Classification) Full-Screen with Scroll -->
  <!-- This markup remains hidden in the original window and will be injected into the new tab -->
  <div id="screen2">
    <h2 style="text-align:center;">AMS view: Ticket classification and duplicate detection</h2>
    <!-- Ticket ID Field (auto-generated) -->
    <div class="form-group">
      <label for="ticketID">Ticket ID:</label>
      <input type="text" id="ticketID" name="ticketID" readonly>
    </div>
    <!-- Fields from Screen 1 (pre-filled) + Classify Ticket Button -->
    <div class="form-group">
      <label for="summary2">Summary:</label>
      <input type="text" id="summary2" name="summary2">
    </div>
    <div class="form-group">
      <label for="description2">Description:</label>
      <textarea id="description2" name="description2" rows="10"></textarea>
    </div>
    <!-- Attachments (placeholder) -->
    <div class="form-group">
      <label for="attachments2">Attachments:</label>
      <input type="file" id="attachments2" name="attachments2" multiple>
    </div>
    <!-- Status (always open) -->
    <div class="form-group">
      <label for="statusField">Status:</label>
      <input type="text" id="statusField" readonly>
    </div>
    <!-- User Priority (passed from Screen 1) -->
    <div class="form-group">
      <label for="priority2">User Priority:</label>
      <input type="text" id="priority2" readonly>
    </div>
    <div class="form-group">
      <label for="component2">Component:</label>
      <input type="text" id="component2" name="component2">
    </div>
    <div class="form-group">
      <label for="company_code2">Company:</label>
      <input type="text" id="company_code2" name="company_code2">
    </div>
    <button id="classifyBtn" onclick="classifyTicket()">Classify Ticket</button>
    <!-- Classification Results (hidden until classification is done) -->
    <div id="classificationResults" style="display:none; margin-top:20px;">
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
      <!-- Impact field (auto-derived from urgency) -->
      <div class="form-group" id="impactFieldGroup" style="display:none;">
        <label for="impactField">Impact:</label>
        <input type="text" id="impactField" readonly>
      </div>
      <!-- Duplicate Info -->
      <div id="duplicateInfoSection" style="display:none; margin-top:20px;">
        <div class="form-group">
          <label for="originalTicketId">Duplicate Ticket List:</label>
          <input type="text" id="originalTicketId">
        </div>
        <div class="form-group">
          <label for="similarityScore">Confidence for Duplicates:</label>
          <input type="text" id="similarityScore">
        </div>
      </div>
      <!-- Assignment Group -->
      <div id="assignmentGroupSection" style="display:none; margin-top:20px;">
        <div class="form-group">
          <label for="assignmentGroup">Assignment Group:</label>
          <input type="text" id="assignmentGroup">
        </div>
      </div>
    </div>
    <!-- Debug Section: Collapsible -->
    <button class="toggle-btn" style="display:none;" id="debugToggle" onclick="toggleDebug()">Show Debug Info</button>
    <div class="toggle-content" id="debugSection">
      <h3>Similar Tickets</h3>
      <div id="similar_tickets"></div>
      <h3>Context Relevance</h3>
      <div id="context_relevance"></div>
      <h3>Answer Relevance</h3>
      <div id="answer_relevance"></div>
    </div>
    <!-- Assignment Insights: Collapsible -->
    <button class="toggle-btn" style="display:none;" id="assignmentInsightsToggle" onclick="toggleAssignmentInsights()">Show Assignment Insights</button>
    <div class="toggle-content" id="assignmentInsights">
      <div id="insights_data"></div>
    </div>
    <!-- Back Button -->
    <button style="background-color:#6c757d; width:auto; margin-top:20px;" onclick="goBack()">Back</button>
  </div>

  <script>
    // Generate a random unique ticket id with prefix "REQ-"
    function generateTicketID() {
      const randomNum = Math.floor(100000 + Math.random() * 900000);
      return "REQ-" + randomNum;
    }
    
    /****************************************
     * SCREEN 1: Validate & Create Ticket   *
     ****************************************/
    function validateTicket() {
      const summary = document.getElementById('summary').value.trim();
      const description = document.getElementById('description').value.trim();
      const component = document.getElementById('component').value.trim();
      const company_code = document.getElementById('company_code').value.trim();
      const priority = document.getElementById('priority').value;
      const attachments = document.getElementById('attachments').files;
      
      if (!summary || !description || !component || !company_code) {
        alert('Summary, Description, Company, and Component are mandatory.');
        return;
      }
      
      // Send only the required fields for validation
      fetch('/process_ticket', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          summary,
          description,
          component,
          company_code,
          priority,
          step: "validate"
        })
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
          // Hide Screen 1 in the original window
          document.getElementById('screen1').style.display = 'none';
          // Reset AMS view
          resetAMSView();
          // Pre-fill AMS fields with Screen 1 data
          document.getElementById('statusField').value = "open";
          document.getElementById('ticketID').value = generateTicketID();
          document.getElementById('priority2').value = priority;
          document.getElementById('summary2').value = summary;
          document.getElementById('description2').value = description;
          document.getElementById('component2').value = component;
          document.getElementById('company_code2').value = company_code;
          
          // Open Screen 2 in a new tab (all existing functionality will be available there)
          const newTab = window.open("", "_blank");
          const screen2Content = document.getElementById('screen2').outerHTML;
          newTab.document.open();
          newTab.document.write(`
            <!DOCTYPE html>
            <html>
            <head>
              <title>AMS View: Ticket Classification & Duplicate Detection</title>
              <meta name="viewport" content="width=device-width, initial-scale=1">
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
                input[type="file"],
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
                #screen2 {
                  display: block;
                  position: static;
                  width: auto;
                  height: auto;
                  margin: 0;
                  padding: 20px;
                  background-color: #f8f9fa;
                  overflow-y: auto;
                }
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
                .toggle-btn {
                  background-color: #6c757d;
                  color: white;
                  padding: 10px;
                  border: none;
                  border-radius: 5px;
                  cursor: pointer;
                  width: 100%;
                  margin-top: 10px;
                  text-align: left;
                }
                .toggle-content {
                  display: none;
                  margin-top: 10px;
                }
              </style>
            </head>
            <body>
              ${screen2Content}
              <script>
                // Pre-populate Screen 2 fields (they were set in the original window)
                document.getElementById('statusField').value = "open";
                document.getElementById('ticketID').value = "${generateTicketID()}";
                document.getElementById('priority2').value = "${priority}";
                document.getElementById('summary2').value = "${summary.replace(/"/g, '&quot;')}";
                document.getElementById('description2').value = "${description.replace(/"/g, '&quot;')}";
                document.getElementById('component2').value = "${component.replace(/"/g, '&quot;')}";
                document.getElementById('company_code2').value = "${company_code.replace(/"/g, '&quot;')}";
                
                /****************************************
                 * SCREEN 2 FUNCTIONS (New Tab Context) *
                 ****************************************/
                function classifyTicket() {
                  const summary = document.getElementById('summary2').value.trim();
                  const description = document.getElementById('description2').value.trim();
                  const component = document.getElementById('component2').value.trim();
                  const company_code = document.getElementById('company_code2').value.trim();
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
                      document.getElementById('classifyBtn').style.display = 'block';
                      return;
                    }
                    alert(data.message || "Classification completed.");
                    document.getElementById('classificationResults').style.display = 'block';
                    if (data.classification) {
                      try {
                        const classificationObj = JSON.parse(data.classification);
                        setFieldValueOrHide("incidentType", "incidentTypeGroup", classificationObj.Incident_Type);
                        setFieldValueOrHide("category1", "category1Group", classificationObj.Category_1);
                        setFieldValueOrHide("category2", "category2Group", classificationObj.Category_2);
                        setFieldValueOrHide("category3", "category3Group", classificationObj.Category_3);
                        setFieldValueOrHide("urgencyField", "urgencyFieldGroup", classificationObj.Urgency);
                        deriveImpact(classificationObj.Urgency);
                      } catch (e) {
                        console.warn("Could not parse classification JSON.");
                        setFieldValueOrHide("incidentType", "incidentTypeGroup", data.classification);
                      }
                    }
                    if (data.duplicate_debug && data.duplicate_debug.is_duplicate) {
                      document.getElementById('duplicateInfoSection').style.display = 'block';
                      document.getElementById('originalTicketId').value = data.duplicate_debug.original_ticket_id || "";
                      let confidence = Math.round((data.duplicate_debug.similarity || 0) * 100);
                      document.getElementById('similarityScore').value = confidence.toString();
                    } else {
                      document.getElementById('duplicateInfoSection').style.display = 'none';
                    }
                    if (data.assignment_group) {
                      document.getElementById('assignmentGroupSection').style.display = 'block';
                      document.getElementById('assignmentGroup').value = data.assignment_group;
                    } else {
                      document.getElementById('assignmentGroupSection').style.display = 'none';
                    }
                    document.getElementById('debugToggle').style.display = 'block';
                    document.getElementById('assignmentInsightsToggle').style.display = 'block';
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
                
                function deriveImpact(urgencyVal) {
                  if (!urgencyVal) return;
                  const urgency = urgencyVal.trim().toLowerCase();
                  let impact = "";
                  switch (urgency) {
                    case "emergency": impact = "Disaster"; break;
                    case "high": impact = "High"; break;
                    case "medium": impact = "Medium"; break;
                    case "low": impact = "Low"; break;
                    default: impact = "Unknown";
                  }
                  document.getElementById('impactField').value = impact;
                  document.getElementById('impactFieldGroup').style.display = 'block';
                }
                
                function setFieldValueOrHide(fieldId, groupId, value) {
                  if (value && value.trim() !== "") {
                    document.getElementById(fieldId).value = value;
                    document.getElementById(groupId).style.display = 'block';
                  } else {
                    document.getElementById(groupId).style.display = 'none';
                  }
                }
                
                function toggleDebug() {
                  const debugSection = document.getElementById('debugSection');
                  debugSection.style.display = (debugSection.style.display === 'none' || debugSection.style.display === '') ? 'block' : 'none';
                  document.getElementById('debugToggle').textContent =
                    debugSection.style.display === 'block' ? 'Hide Debug Info' : 'Show Debug Info';
                }
                
                function toggleAssignmentInsights() {
                  const insightsSection = document.getElementById('assignmentInsights');
                  insightsSection.style.display = (insightsSection.style.display === 'none' || insightsSection.style.display === '') ? 'block' : 'none';
                  document.getElementById('assignmentInsightsToggle').textContent =
                    insightsSection.style.display === 'block' ? 'Hide Assignment Insights' : 'Show Assignment Insights';
                  if (insightsSection.style.display === 'block') {
                    fetchInsights();
                  }
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
                  let html = '<table><tr><th>Incident Type</th><th>Category1</th><th>Category2</th><th>Category3</th><th>Urgency</th><th>Confidence for Duplicates</th><th>Cross-Encoder</th><th>Combined Score</th></tr>';
                  tickets.forEach(ticket => {
                    const confidence = Math.round((ticket.Similarity_Score || 0) * 100);
                    html += `<tr>
                      <td>${ticket.Incident_Type}</td>
                      <td>${ticket.Category1}</td>
                      <td>${ticket.Category2}</td>
                      <td>${ticket.Category3}</td>
                      <td>${ticket.Urgency}</td>
                      <td>${confidence}</td>
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
                
                function goBack() {
                  if (window.opener && !window.opener.closed) {
                    window.opener.document.getElementById('screen1').style.display = 'block';
                  }
                  window.close();
                }
                
                // Bind functions to the buttons in the new tab
                document.getElementById('classifyBtn').onclick = classifyTicket;
                document.querySelector('button[onclick="goBack()"]').onclick = goBack;
              <\/script>
            </body>
            </html>
          `);
          newTab.document.close();
        }, 2000);
      })
      .catch(error => {
        console.error('Error:', error);
        alert('Error: ' + error.message);
      });
    }
    
    /****************************************
     * SCREEN 2: Classify Ticket & Display  *
     ****************************************/
    function classifyTicket() {
      // This fallback function in the original window won't be used when new tab is open.
      alert("Classification is handled in the new tab.");
    }
    
    /****************************************
     * Helper Functions (Reset, etc.)       *
     ****************************************/
    function resetAMSView() {
      document.getElementById('statusField').value = "";
      document.getElementById('ticketID').value = "";
      document.getElementById('priority2').value = "";
      document.getElementById('attachments2').value = "";
      document.getElementById('classificationResults').style.display = 'none';
      document.getElementById('incidentTypeGroup').style.display = 'none';
      document.getElementById('category1Group').style.display = 'none';
      document.getElementById('category2Group').style.display = 'none';
      document.getElementById('category3Group').style.display = 'none';
      document.getElementById('urgencyFieldGroup').style.display = 'none';
      document.getElementById('impactFieldGroup').style.display = 'none';
      document.getElementById('duplicateInfoSection').style.display = 'none';
      document.getElementById('assignmentGroupSection').style.display = 'none';
      document.getElementById('debugToggle').style.display = 'none';
      document.getElementById('assignmentInsightsToggle').style.display = 'none';
      document.getElementById('debugSection').style.display = 'none';
      document.getElementById('assignmentInsights').style.display = 'none';
      document.getElementById('classifyBtn').style.display = 'block';
      document.getElementById('summary2').value = "";
      document.getElementById('description2').value = "";
      document.getElementById('component2').value = "";
      document.getElementById('company_code2').value = "";
      document.getElementById('incidentType').value = "";
      document.getElementById('category1').value = "";
      document.getElementById('category2').value = "";
      document.getElementById('category3').value = "";
      document.getElementById('urgencyField').value = "";
      document.getElementById('impactField').value = "";
    }
    
    // Back button in the original window (for fallback)
    function goBack() {
      document.getElementById('screen2').style.display = 'none';
      document.getElementById('screen1').style.display = 'block';
      resetAMSView();
    }
  </script>
</body>
</html>
