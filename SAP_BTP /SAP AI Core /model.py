def get_token():
    """Retrieve the authentication token using client credentials."""
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
    encoded_auth = base64.b64encode(auth_string.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded_auth}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(AUTH_URL, headers=headers, data=data)
    if response.status_code == 200:
        print("Authentication successful")
        return response.json()["access_token"]
    else:
        print(f"Authentication failed: {response.status_code}\n{response.text}")
        return None

def translate_document(file_path, source_lang="es-ES", target_lang="en-US"):
    """Translate document content using POST /api/v1/translation."""
    token = get_token()
    if not token:
        return None

    # Read the file content
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept-Language": "en"  # Localize error messages in English
    }
    
    # Updated payload with singular targetLanguage field and the file content under "data"
    payload = {
        "sourceLanguage": source_lang,
        "targetLanguage": target_lang,
        "data": content
    }

    url = f"{API_BASE_URL}/api/v1/translation"
    print(f"Trying endpoint: {url}")
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        translated_text = result.get("data")
        if not translated_text:
            print("Translation succeeded but no translated data found in response.")
            return None
        output_file = file_path.replace(".txt", f"_{target_lang}.txt")
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(translated_text)
        print(f"Translated file saved as: {output_file}")
        return output_file
    else:
        print(f"Translation failed with status: {response.status_code}\n{response.text}")
        return None

if __name__ == "__main__":
    file_path = "spanish.txt"  # Replace with your file name
    translate_document(file_path)
