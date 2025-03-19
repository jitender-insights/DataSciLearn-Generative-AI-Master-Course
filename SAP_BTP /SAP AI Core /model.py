<!DOCTYPE html>
<html>
<head>
    <title>JIRA Ticket Classification & Duplicate Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }
        h1, h2, h3 { text-align: center; color: #333; }
        .form-group { margin-bottom: 15px; }
        label { font-weight: bold; }
        input[type="text"], textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { background-color: #007bff; color: white; padding: 12px 18px; border: none; border-radius: 5px; cursor: pointer; width: 100%; margin-top: 10px; }
        #result { margin-top: 20px; padding: 15px; border-radius: 5px; background-color: #ffffff; box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.1); display: none; }
        .bold-message { font-weight: bold; color: #d9534f; text-align: center; margin-top: 10px; font-size: 18px; }
        .debug-toggle { margin-top: 10px; background-color: #28a745; color: white; width: 100%; padding: 10px; border: none; border-radius: 5px; cursor: pointer; }
        #debug-section { display: none; margin-top: 15px; padding: 10px; background-color: #f1f1f1; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 14px; }
        th { background-color: #007bff; color: #fff; }
        .progress-bar { height: 20px; background-color: #e0e0e0; border-radius: 10px; margin-top: 5px; overflow: hidden; width: 100px; }
        .progress { height: 100%; background-color: #4CAF50; text-align: center; line-height: 20px; color: white; }
        #insights { margin-top: 30px; }
        #insights table { margin: 0 auto; }
        /* Hide second screen initially */
        #screen2 { display: none; }
    </style>
</head>
<body>
    <h1>JIRA Ticket Classification & Duplicate Detection</h1>
    
    <!-- Screen 1: Ticket Validation -->
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
        <button onclick="validateTicket()">Validate Ticket</button>
    </div>
    
    <!-- Screen 2: AMS Screen (with extra options) -->
    <div id="screen2">
        <h2>AMS Ticket Options</h2>
        <!-- Pre-filled fields from Screen 1 -->
        <div class="form-group">
            <label for="summary2">Summary:</label>
            <input type="text" id="summary2" name="summary2" readonly>
        </div>
        <div class="form-group">
            <label for="description2">Description:</label>
            <textarea id="description2" name="description2" rows="10" readonly></textarea>
        </div>
        <div class="form-group">
            <label for="component2">Component:</label>
            <input type="text" id="component2" name="component2" readonly>
        </div>
        <div class="form-group">
            <label for="company_code2">Company Code:</label>
            <input type="text" id="company_code2" name="company_code2" readonly>
        </div>
        <!-- Additional fields -->
        <div class="form-group">
            <label for="urgency">Urgency:</label>
            <select id="urgency" name="urgency">
                <option value="">--Select Option--</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
            </select>
        </div>
        <div class="form-group">
            <label for="labels">Labels:</label>
            <input type="text" id="labels" name="labels" placeholder="Enter labels">
        </div>
        <div class="form-group">
            <label for="jira_ticket_option">JIRA Ticket Option:</label>
            <select id="jira_ticket_option" name="jira_ticket_option">
                <option value="">--Select Option--</option>
                <option value="option1">Option 1</option>
                <option value="option2">Option 2</option>
            </select>
        </div>
        <button onclick="submitTicket()">Submit Ticket</button>
    </div>
    
    <!-- Result & Debug Section (kept as-is) -->
    <div id="result">
        <h2>Result</h2>
        <div id="classification"></div>
        <p class="bold-message" id="assignment_group"></p>
        <p class="bold-message" id="message"></p>
        <button class="debug-toggle" onclick="toggleDebug()">Show Debug Info</button>
        <div id="debug-section">
            <h3>Duplicate Debug</h3>
            <div id="duplicate_debug"></div>
            <h3>Similar Tickets</h3>
            <div id="similar_tickets"></div>
            <h3>Context Relevance</h3>
            <div id="context_relevance"></div>
            <h3>Answer Relevance</h3>
            <div id="answer_relevance"></div>
        </div>
    </div>
    
    <!-- Insights Section -->
    <div id="insights">
        <h2>Agent Insights</h2>
        <button onclick="fetchInsights()">Refresh Insights</button>
        <div id="insights_data"></div>
    </div>
    
    <script>
        // Validate Ticket Info (Step 1)
        function validateTicket() {
            const summary = document.getElementById('summary').value.trim();
            const description = document.getElementById('description').value.trim();
            const component = document.getElementById('component').value.trim();
            const company_code = document.getElementById('company_code').value.trim();
            
            if (!summary || !description || !component || !company_code) {
                alert('Summary, Description, Company Code, and Component are mandatory.');
                return;
            }
            
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
                // Ticket information is sufficient – move to AMS screen
                document.getElementById('screen1').style.display = 'none';
                document.getElementById('screen2').style.display = 'block';
                // Pre-fill AMS screen fields
                document.getElementById('summary2').value = summary;
                document.getElementById('description2').value = description;
                document.getElementById('component2').value = component;
                document.getElementById('company_code2').value = company_code;
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
        
        // Submit Ticket with Additional AMS Info (Step 2)
        function submitTicket() {
            // Collect basic info (from read-only fields) and extra AMS fields
            const summary = document.getElementById('summary2').value.trim();
            const description = document.getElementById('description2').value.trim();
            const component = document.getElementById('component2').value.trim();
            const company_code = document.getElementById('company_code2').value.trim();
            const urgency = document.getElementById('urgency').value.trim();
            const labels = document.getElementById('labels').value.trim();
            const jira_ticket_option = document.getElementById('jira_ticket_option').value.trim();
            
            fetch('/process_ticket', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ summary, description, component, company_code, urgency, labels, jira_ticket_option, step: "final" })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                document.getElementById('result').style.display = 'block';
                document.getElementById('message').innerText = data.message || "";
                document.getElementById('assignment_group').innerText = "Assignment Group: " + (data.assignment_group || "N/A");
                
                // Process additional data for classification, duplicate debug, etc.
                if (data.subtask) {
                    let subtaskObj = data.subtask;
                    let subtaskHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                    for (const [key, val] of Object.entries(subtaskObj)) {
                        subtaskHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                    }
                    subtaskHTML += '</table>';
                    document.getElementById('classification').innerHTML = subtaskHTML;
                    if (data.duplicate_debug) {
                        let dupDebugHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                        for (const [key, val] of Object.entries(data.duplicate_debug)) {
                            dupDebugHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                        }
                        dupDebugHTML += '</table>';
                        document.getElementById('duplicate_debug').innerHTML = dupDebugHTML;
                    }
                } else if (data.classification) {
                    buildClassificationTable(data.classification, 'classification');
                    buildSimilarTicketsTable(data.similar_tickets, 'similar_tickets');
                    buildContextRelevanceTable(data.context_relevance, 'context_relevance');
                    buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
                    if (data.duplicate_debug) {
                        let dupDebugHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                        for (const [key, val] of Object.entries(data.duplicate_debug)) {
                            dupDebugHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                        }
                        dupDebugHTML += '</table>';
                        document.getElementById('duplicate_debug').innerHTML = dupDebugHTML;
                    }
                }
                // Refresh insights after processing a ticket
                fetchInsights();
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
        
        function toggleDebug() {
            const debugSection = document.getElementById('debug-section');
            debugSection.style.display = (debugSection.style.display === 'none' || debugSection.style.display === '') ? 'block' : 'none';
        }
        
        function buildClassificationTable(classificationText, containerId) {
            let classificationObj;
            try {
                classificationObj = JSON.parse(classificationText);
            } catch(e) {
                document.getElementById(containerId).innerText = classificationText;
                return;
            }
            if (!classificationObj) {
                document.getElementById(containerId).innerText = 'No classification data.';
                return;
            }
            let html = '<table><tr><th>Field</th><th>Value</th></tr>';
            for (const [key, val] of Object.entries(classificationObj)) {
                html += `<tr><td>${key}</td><td>${val}</td></tr>`;
            }
            html += '</table>';
            document.getElementById(containerId).innerHTML = html;
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
        
        // Fetch insights from /chart_data endpoint and display them
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
    </script>
</body>
</html>
