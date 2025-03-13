<!DOCTYPE html>
<html>
<head>
    <title>JIRA Ticket Classification & Duplicate Detection</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; }
        h1 { text-align: center; color: #333; }
        .form-group { margin-bottom: 15px; }
        label { font-weight: bold; }
        input[type="text"], textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
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
    <button onclick="processTicket()">Process Ticket</button>
   
    <!-- New section for duplicate decision drop down -->
    <div id="duplicate_decision_section" style="display:none; margin-top:20px;">
        <h2>Please select duplicate decision</h2>
        <select id="duplicate_decision_select"></select>
        <button onclick="submitDuplicateDecision()">Submit Duplicate Decision</button>
    </div>
   
    <!-- New section for classification decision drop down -->
    <div id="classification_decision_section" style="display:none; margin-top:20px;">
        <h2>Please select classification decision</h2>
        <select id="classification_decision_select"></select>
        <button onclick="submitClassificationDecision()">Submit Classification Decision</button>
    </div>
   
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
        // Global variable to store the "ticket_data" from the server
        let globalTicketData = null;
 
        function processTicket() {
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
                body: JSON.stringify({ summary, description, component, company_code })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                    return;
                }
                // Store ticket_data in our global variable
                if (data.ticket_data) {
                    globalTicketData = data.ticket_data;
                }
               
                document.getElementById('result').style.display = 'block';
                document.getElementById('message').innerText = data.message || '';
               
                if (data.action && data.action === 'duplicate_decision') {
                    // Show the duplicate decision section
                    document.getElementById('duplicate_decision_section').style.display = 'block';
                    const dupSelect = document.getElementById('duplicate_decision_select');
                    dupSelect.innerHTML = '';
                    data.options.forEach(opt => {
                        const optionElem = document.createElement('option');
                        optionElem.value = opt;
                        optionElem.text = opt;
                        dupSelect.appendChild(optionElem);
                    });
                }
                else if (data.action && data.action === 'classification_decision') {
                    // Show the classification decision section
                    document.getElementById('classification_decision_section').style.display = 'block';
                    const classSelect = document.getElementById('classification_decision_select');
                    classSelect.innerHTML = '';
                    data.options.forEach(opt => {
                        const optionElem = document.createElement('option');
                        optionElem.value = opt;
                        optionElem.text = opt;
                        classSelect.appendChild(optionElem);
                    });
                    // Build debug and other info if available
                    if (data.duplicate_debug) {
                        buildDebugInfo(data.duplicate_debug, 'duplicate_debug');
                    }
                    if (data.classification) {
                        buildClassificationTable(data.classification, 'classification');
                    }
                    if (data.similar_tickets) {
                        buildSimilarTicketsTable(data.similar_tickets, 'similar_tickets');
                    }
                    if (data.context_relevance) {
                        buildContextRelevanceTable(data.context_relevance, 'context_relevance');
                    }
                    if (data.answer_relevance) {
                        buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
                    }
                }
                else {
                    // No special action: either it's a subtask or a final classification
                    if (data.subtask) {
                        let subtaskObj = data.subtask;
                        let subtaskHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                        for (const [key, val] of Object.entries(subtaskObj)) {
                            subtaskHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                        }
                        subtaskHTML += '</table>';
                        document.getElementById('classification').innerHTML = subtaskHTML;
                        if (data.duplicate_debug) {
                            buildDebugInfo(data.duplicate_debug, 'duplicate_debug');
                        }
                    }
                    else if (data.classification) {
                        buildClassificationTable(data.classification, 'classification');
                        if (data.duplicate_debug) {
                            buildDebugInfo(data.duplicate_debug, 'duplicate_debug');
                        }
                        if (data.similar_tickets) {
                            buildSimilarTicketsTable(data.similar_tickets, 'similar_tickets');
                        }
                        if (data.context_relevance) {
                            buildContextRelevanceTable(data.context_relevance, 'context_relevance');
                        }
                        if (data.answer_relevance) {
                            buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
                        }
                    }
                    // Optionally reload page after a few seconds
                    // setTimeout(function(){ location.reload(); }, 3000);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
       
        function submitDuplicateDecision() {
            const decision = document.getElementById('duplicate_decision_select').value;
            const summary = document.getElementById('summary').value.trim();
            const description = document.getElementById('description').value.trim();
            const component = document.getElementById('component').value.trim();
            const company_code = document.getElementById('company_code').value.trim();
           
            // Pass globalTicketData forward with the request
            fetch('/duplicate_decision', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    decision,
                    summary,
                    description,
                    component,
                    company_code,
                    ticket_data: globalTicketData
                })
            })
            .then(response => response.json())
            .then(data => {
                // If subtask data is returned (i.e. user accepted duplicate)
                if (data.subtask) {
                    document.getElementById('result').style.display = 'block';
                    let subtaskObj = data.subtask;
                    let subtaskHTML = '<table><tr><th>Field</th><th>Value</th></tr>';
                    for (const [key, val] of Object.entries(subtaskObj)) {
                        subtaskHTML += `<tr><td>${key}</td><td>${val}</td></tr>`;
                    }
                    subtaskHTML += '</table>';
                    document.getElementById('classification').innerHTML = subtaskHTML;
                    if (data.duplicate_debug) {
                        buildDebugInfo(data.duplicate_debug, 'duplicate_debug');
                    }
                    // Hide the duplicate decision section after processing
                    document.getElementById('duplicate_decision_section').style.display = 'none';
                } else if (data.classification) {
                    // For "reject duplicate" scenario where classification is returned
                    document.getElementById('result').style.display = 'block';
                    buildClassificationTable(data.classification, 'classification');
                    if (data.similar_tickets) {
                        buildSimilarTicketsTable(data.similar_tickets, 'similar_tickets');
                    }
                    if (data.context_relevance) {
                        buildContextRelevanceTable(data.context_relevance, 'context_relevance');
                    }
                    if (data.answer_relevance) {
                        buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
                    }
                    document.getElementById('message').innerText = data.message || '';
                    document.getElementById('duplicate_decision_section').style.display = 'none';
                } else {
                    alert(data.message);
                    setTimeout(function(){ location.reload(); }, 3000);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            });
        }
       
        function submitClassificationDecision() {
            const decision = document.getElementById('classification_decision_select').value;
            const summary = document.getElementById('summary').value.trim();
            const description = document.getElementById('description').value.trim();
            const component = document.getElementById('component').value.trim();
            const company_code = document.getElementById('company_code').value.trim();
           
            fetch('/classification_decision', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    decision,
                    summary,
                    description,
                    component,
                    company_code,
                    ticket_data: globalTicketData
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.classification) {
                    document.getElementById('result').style.display = 'block';
                    buildClassificationTable(data.classification, 'classification');
                    if (data.context_relevance) {
                        buildContextRelevanceTable(data.context_relevance, 'context_relevance');
                    }
                    if (data.answer_relevance) {
                        buildAnswerRelevanceTable(data.answer_relevance, 'answer_relevance');
                    }
                    document.getElementById('message').innerText = data.message || '';
                    document.getElementById('classification_decision_section').style.display = 'none';
                } else {
                    alert(data.message);
                    setTimeout(function(){ location.reload(); }, 3000);
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
 
        // Helper functions to build tables
        function buildDebugInfo(debugData, containerId) {
            let html = '<table><tr><th>Field</th><th>Value</th></tr>';
            for (const [key, val] of Object.entries(debugData)) {
                html += `<tr><td>${key}</td><td>${val}</td></tr>`;
            }
            html += '</table>';
            document.getElementById(containerId).innerHTML = html;
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
