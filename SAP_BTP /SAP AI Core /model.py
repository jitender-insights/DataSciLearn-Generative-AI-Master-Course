<!DOCTYPE html>
<html>
<head>
    <title>JIRA Ticket Classification & Duplicate Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }
        h1 { text-align: center; color: #333; }
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
    </style>
</head>
<body>
    <h1>JIRA Ticket Classification & Duplicate Detection</h1>
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
    <!-- Duplicate Decision dropdown (hidden by default) -->
    <div class="form-group" id="duplicate_decision_group" style="display:none;">
        <label for="duplicate_decision">Duplicate Decision:</label>
        <select id="duplicate_decision" name="duplicate_decision">
            <option value="">--Select Option--</option>
            <option value="accept duplicate">Accept Duplicate</option>
            <option value="reject duplicate">Reject Duplicate</option>
        </select>
    </div>
    <!-- Classification Decision dropdown (hidden by default) -->
    <div class="form-group" id="classification_decision_group" style="display:none;">
        <label for="classification_decision">Classification Decision:</label>
        <select id="classification_decision" name="classification_decision">
            <option value="">--Select Option--</option>
            <option value="accept classification">Accept Classification</option>
            <option value="reject classification">Reject Classification</option>
        </select>
    </div>
    <button onclick="processTicket()">Process Ticket</button>
    <div id="result">
        <h2>Result</h2>
        <div id="classification"></div>
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
    <script>
        function processTicket() {
            const summary = document.getElementById('summary').value.trim();
            const description = document.getElementById('description').value.trim();
            const component = document.getElementById('component').value.trim();
            const company_code = document.getElementById('company_code').value.trim();
            // Retrieve values from dropdowns only if visible
            const duplicateDecisionElem = document.getElementById('duplicate_decision');
            const classificationDecisionElem = document.getElementById('classification_decision');
            const duplicate_decision = duplicateDecisionElem ? duplicateDecisionElem.value.trim() : "";
            const classification_decision = classificationDecisionElem ? classificationDecisionElem.value.trim() : "";
            
            if (!summary || !description || !component || !company_code) {
                alert('Summary, Description, Company Code, and Component are mandatory.');
                return;
            }
            fetch('/process_ticket', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ summary, description, component, company_code, duplicate_decision, classification_decision })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                document.getElementById('result').style.display = 'block';
                document.getElementById('message').innerText = data.message || '';

                // Based on the returned message, decide whether to show dropdowns:
                const msg = data.message.toLowerCase();
                // Show duplicate decision dropdown only if awaiting duplicate decision
                if (msg.includes("awaiting agent decision") && msg.includes("duplicate")) {
                    document.getElementById('duplicate_decision_group').style.display = 'block';
                } else {
                    document.getElementById('duplicate_decision_group').style.display = 'none';
                }
                // Show classification decision dropdown only if awaiting classification decision
                if (msg.includes("awaiting agent decision") && msg.includes("classification")) {
                    document.getElementById('classification_decision_group').style.display = 'block';
                } else {
                    document.getElementById('classification_decision_group').style.display = 'none';
                }
                
                // If a duplicate is detected:
                if (data.subtask) {
                    // Show the subtask JSON in the classification section
                    let subtaskObj = data.subtask;
                    let subtaskHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                    for (const [key, val] of Object.entries(subtaskObj)) {
                        subtaskHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                    }
                    subtaskHTML += '</table>';
                    document.getElementById('classification').innerHTML = subtaskHTML;
                    
                    // Display duplicate debug info in the debug section
                    if (data.duplicate_debug) {
                        let dupDebugHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                        for (const [key, val] of Object.entries(data.duplicate_debug)) {
                            dupDebugHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                        }
                        dupDebugHTML += '</table>';
                        document.getElementById('duplicate_debug').innerHTML = dupDebugHTML;
                    }
                }
                // If not a duplicate, show classification & debug as before
                else if (data.classification) {
                    buildClassificationTable(data.classification, 'classification');
                    buildSimilarTicketsTable(data.similar_tickets, 'similar_tickets');
                    buildContextRelevanceTable(data.context_relevance, 'context_relevance');
                    buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
                    // Display duplicate debug info if available
                    if (data.duplicate_debug) {
                        let dupDebugHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                        for (const [key, val] of Object.entries(data.duplicate_debug)) {
                            dupDebugHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                        }
                        dupDebugHTML += '</table>';
                        document.getElementById('duplicate_debug').innerHTML = dupDebugHTML;
                    }
                }
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
    </script>
</body>
</html>
