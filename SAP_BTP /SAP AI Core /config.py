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
            display: none; /* Hidden by default */
            position: fixed;
            z-index: 999;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            overflow: auto;
            background-color: rgba(0,0,0,0.4); /* Semi-transparent background */
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
        /* Classification result table styles */
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
        /* Debug section */
        #debug-section {
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
    </style>
</head>
<body>
    <h1>JIRA Ticket Classification & Duplicate Detection</h1>
    
    <!-- Screen 1: Ticket Creation (User Input) -->
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
            <label for="company_code">Company:</label>
            <input type="text" id="company_code" name="company_code">
        </div>
        <div class="form-group">
            <label for="priority">Priority:</label>
            <select id="priority" name="priority">
                <option value="">--Select Priority--</option>
                <option value="low">Low</option>
                <option value="medium">Medium</option>
                <option value="high">High</option>
            </select>
        </div>
        <button onclick="validateTicket()">Create Ticket</button>
    </div>
    
    <!-- Loader Modal (appears after successful validation) -->
    <div id="loaderModal" class="modal">
      <div class="modal-content">
        <div class="loader"></div>
        <p>Ticket is sufficient for classification. Please wait...</p>
      </div>
    </div>
    
    <!-- Screen 2: AMS (Classification) -->
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
            <label for="company_code2">Company:</label>
            <input type="text" id="company_code2" name="company_code2" readonly>
        </div>
        <div class="form-group">
            <label for="priority2">Priority:</label>
            <input type="text" id="priority2" name="priority2" readonly>
        </div>
        <!-- Classify Ticket button -->
        <button onclick="classifyTicket()">Classify Ticket</button>
        
        <!-- Classification Result Section -->
        <div id="classificationResult" style="margin-top:20px;"></div>
        
        <!-- Agent Insights (shown only in Screen 2) -->
        <div id="agentInsights" style="margin-top:30px;">
            <h2>Agent Insights</h2>
            <button onclick="fetchInsights()">Refresh Insights</button>
            <div id="insights_data"></div>
        </div>
        
        <!-- Debug Info Section -->
        <div id="debug-section">
            <button class="debug-toggle" onclick="toggleDebug()">Show Debug Info</button>
            <div id="debugContent" style="display:none;">
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
    </div>
    
    <script>
        // Screen 1: Validate Ticket Info & Create Ticket
        function validateTicket() {
            const summary = document.getElementById('summary').value.trim();
            const description = document.getElementById('description').value.trim();
            const component = document.getElementById('component').value.trim();
            const company_code = document.getElementById('company_code').value.trim();
            const priority = document.getElementById('priority').value.trim();
            
            if (!summary || !description || !component || !company_code || !priority) {
                alert('Summary, Description, Company, Component, and Priority are mandatory.');
                return;
            }
            
            fetch('/process_ticket', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ summary, description, component, company_code, priority, step: "validate" })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error || data.missing_fields) {
                    alert(data.message || "Missing required ticket information.");
                    return;
                }
                // Show loader modal upon successful validation
                document.getElementById('loaderModal').style.display = 'block';
                // After a 2-second delay, hide loader and transition to Screen 2
                setTimeout(() => {
                    document.getElementById('loaderModal').style.display = 'none';
                    document.getElementById('screen1').style.display = 'none';
                    document.getElementById('screen2').style.display = 'block';
                    // Pre-fill Screen 2 fields with Screen 1 data
                    document.getElementById('summary2').value = summary;
                    document.getElementById('description2').value = description;
                    document.getElementById('component2').value = component;
                    document.getElementById('company_code2').value = company_code;
                    document.getElementById('priority2').value = priority;
                }, 2000);
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
        
        // Screen 2: Classify Ticket & Display Results
        function classifyTicket() {
            const summary = document.getElementById('summary2').value.trim();
            const description = document.getElementById('description2').value.trim();
            const component = document.getElementById('component2').value.trim();
            const company_code = document.getElementById('company_code2').value.trim();
            const priority = document.getElementById('priority2').value.trim();
            
            fetch('/process_ticket', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ summary, description, component, company_code, priority, step: "final" })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                let resultHTML = `<p>${data.message || ""}</p>`;
                resultHTML += `<p><strong>Assignment Group:</strong> ${data.assignment_group || "N/A"}</p>`;
                
                if (data.classification) {
                    resultHTML += `<h3>Classification</h3>`;
                    try {
                        const classificationObj = JSON.parse(data.classification);
                        resultHTML += '<table><tr><th>Field</th><th>Value</th></tr>';
                        for (const [key, val] of Object.entries(classificationObj)) {
                            resultHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                        }
                        resultHTML += '</table>';
                    } catch(e) {
                        resultHTML += `<p>${data.classification}</p>`;
                    }
                }
                if (data.duplicate_debug) {
                    resultHTML += `<h3>Duplicate Debug</h3>`;
                    resultHTML += '<table><tr><th>Field</th><th>Value</th></tr>';
                    for (const [key, val] of Object.entries(data.duplicate_debug)) {
                        resultHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                    }
                    resultHTML += '</table>';
                }
                if (data.similar_tickets) {
                    resultHTML += `<h3>Similar Tickets</h3>`;
                    let html = '<table><tr><th>Incident Type</th><th>Category1</th><th>Category2</th><th>Category3</th><th>Priority</th><th>Similarity</th><th>Cross-Encoder</th><th>Combined Score</th></tr>';
                    data.similar_tickets.forEach(ticket => {
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
                    resultHTML += html;
                }
                if (data.context_relevance) {
                    resultHTML += `<h3>Context Relevance</h3>`;
                    let html = '<table><tr><th>Metric</th><th>Value</th><th>Visualization</th></tr>';
                    for (const [key, val] of Object.entries(data.context_relevance)) {
                        if (typeof val === 'number') {
                            const pct = Math.round(val * 100);
                            html += `<tr><td>${key}</td><td>${val}</td><td><div class="progress-bar"><div class="progress" style="width:${pct}%">${pct}%</div></div></td></tr>`;
                        } else {
                            html += `<tr><td>${key}</td><td colspan="2">${val}</td></tr>`;
                        }
                    }
                    html += '</table>';
                    resultHTML += html;
                }
                if (data.answer_relevance) {
                    resultHTML += `<h3>Answer Relevance</h3>`;
                    let html = '<table><tr><th>Metric</th><th>Value</th><th>Visualization</th></tr>';
                    for (const [key, val] of Object.entries(data.answer_relevance)) {
                        if (typeof val === 'number') {
                            const pct = Math.round(val * 100);
                            html += `<tr><td>${key}</td><td>${val}</td><td><div class="progress-bar"><div class="progress" style="width:${pct}%">${pct}%</div></div></td></tr>`;
                        } else {
                            html += `<tr><td>${key}</td><td colspan="2">${val}</td></tr>`;
                        }
                    }
                    html += '</table>';
                    resultHTML += html;
                }
                document.getElementById('classificationResult').innerHTML = resultHTML;
                // Refresh agent insights after classification
                fetchInsights();
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
        
        // Toggle Debug Information
        function toggleDebug() {
            const debugContent = document.getElementById('debugContent');
            debugContent.style.display = (debugContent.style.display === 'none' || debugContent.style.display === '') ? 'block' : 'none';
        }
        
        // Fetch Agent Insights (only in Screen 2)
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
