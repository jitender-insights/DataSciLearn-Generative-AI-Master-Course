<!DOCTYPE html>
<html>
<head>
  <title>JIRA Ticket Classification & Duplicate Detection</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    /* Shared Styles for Screen 1 and Screen 2 */
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
    /* Screen 2 (AMS View) is hidden in the original window */
    #screen2 {
      display: none;
    }
  </style>
</head>
<body>
  <!-- Screen 1: Ticket Creation -->
  <div id="screen1">
    <h1 id="screen1Header">User Ticket Creation View</h1>
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
      <p>Processing ticket…</p>
    </div>
  </div>

  <!-- Hidden Screen 2: AMS (Classification) View -->
  <div id="screen2">
    <h2 style="text-align:center;">AMS View: Ticket Classification & Duplicate Detection</h2>
    <!-- Ticket ID Field (auto-generated) -->
    <div class="form-group">
      <label for="ticketID">Ticket ID:</label>
      <input type="text" id="ticketID" name="ticketID" readonly>
    </div>
    <!-- Pre-filled Fields from Screen 1 -->
    <div class="form-group">
      <label for="summary2">Summary:</label>
      <input type="text" id="summary2" name="summary2">
    </div>
    <div class="form-group">
      <label for="description2">Description:</label>
      <textarea id="description2" name="description2" rows="10"></textarea>
    </div>
    <div class="form-group">
      <label for="attachments2">Attachments:</label>
      <input type="file" id="attachments2" name="attachments2" multiple>
    </div>
    <!-- Status and Priority -->
    <div class="form-group">
      <label for="statusField">Status:</label>
      <input type="text" id="statusField" readonly>
    </div>
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
      <div class="form-group" id="impactFieldGroup" style="display:none;">
        <label for="impactField">Impact:</label>
        <input type="text" id="impactField" readonly>
      </div>
      <!-- (Additional sections such as duplicate info can be added here) -->
    </div>
    <!-- Back Button -->
    <button style="background-color:#6c757d; width:auto; margin-top:20px;" onclick="goBack()">Back</button>
  </div>

  <script>
    // Generate a unique Ticket ID with prefix "REQ-"
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
      // Attachments placeholder (not processed in this demo)
      const attachments = document.getElementById('attachments').files;
      
      if (!summary || !description || !component || !company_code) {
        alert('Summary, Description, Company, and Component are mandatory.');
        return;
      }
      
      // Example: Send data for validation (adjust URL/payload as needed)
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
        // Show loader modal briefly
        document.getElementById('loaderModal').style.display = 'block';
        setTimeout(() => {
          document.getElementById('loaderModal').style.display = 'none';
          // Hide Screen 1 in the original window
          document.getElementById('screen1').style.display = 'none';
          // (Optionally, reset AMS view here if needed)
          // Open Screen 2 in a new tab using _blank
          const newTab = window.open("", "_blank");
          const screen2Content = document.getElementById('screen2').outerHTML;
          
          // Build the complete HTML for the new tab
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
                h1, h2, h3 { text-align: center; color: #333; }
                .form-group { margin-bottom: 15px; }
                label { font-weight: bold; }
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
              </style>
            </head>
            <body>
              ${screen2Content}
              <script>
                // Pre-populate Screen 2 fields using Screen 1 data
                document.getElementById('statusField').value = "open";
                document.getElementById('ticketID').value = "${generateTicketID()}";
                document.getElementById('priority2').value = "${priority}";
                document.getElementById('summary2').value = "${summary.replace(/"/g, '&quot;')}";
                document.getElementById('description2').value = "${description.replace(/"/g, '&quot;')}";
                document.getElementById('component2').value = "${component.replace(/"/g, '&quot;')}";
                document.getElementById('company_code2').value = "${company_code.replace(/"/g, '&quot;')}";
                
                /****************************************
                 * SCREEN 2 FUNCTIONS (New Tab)         *
                 ****************************************/
                function classifyTicket() {
                  const summary = document.getElementById('summary2').value.trim();
                  const description = document.getElementById('description2').value.trim();
                  const component = document.getElementById('component2').value.trim();
                  const company_code = document.getElementById('company_code2').value.trim();
                  
                  // Hide the classify button to prevent multiple clicks
                  document.getElementById('classifyBtn').style.display = 'none';
                  
                  // Example: Send data for classification (adjust URL/payload as needed)
                  fetch('/process_ticket', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      summary, description, component, company_code, step: "final"
                    })
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
                    // (Additional logic to display classification fields can be added here)
                  })
                  .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('classifyBtn').style.display = 'block';
                    alert('Error: ' + error.message);
                  });
                }
                
                function goBack() {
                  // If the original window is still open, show Screen 1 there
                  if (window.opener && !window.opener.closed) {
                    window.opener.document.getElementById('screen1').style.display = 'block';
                  }
                  window.close();
                }
                
                // Bind functions to buttons in the new tab
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
      // This function remains for the original window (if needed for fallback)
      // When using the new tab, the new tab's classifyTicket is used.
      alert("This function is now handled in the new tab.");
    }
    
    /****************************************
     * Helper Functions (Reset, etc.)       *
     ****************************************/
    function resetAMSView() {
      // Reset fields and hide classification sections in Screen 2
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
    
    // The goBack() function below is for use in the original window (if needed)
    function goBack() {
      document.getElementById('screen2').style.display = 'none';
      document.getElementById('screen1').style.display = 'block';
      resetAMSView();
    }
  </script>
</body>
</html>
