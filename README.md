# Project Verification System

This system verifies projects based on their source code, evaluating originality, complexity, and checking for plagiarism.

## Setup

1. Install MongoDB and start the MongoDB service.

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set environment variables:
   - `GITHUB_TOKEN`: Your GitHub personal access token
   - `MONGODB_URI`: MongoDB connection string (default: 'mongodb://localhost:27017/')

4. Run the Flask backend:
   ```
   python backend/app.py
   ```

5. Run the Streamlit frontend:
   ```
   streamlit run frontend/app.py
   ```

## Usage

1. Prepare a CSV file with columns "Roll.No." and "Github link".
2. Upload the CSV file in the Streamlit interface.
3. Click "Verify Projects" to process and view results.

## API Documentation

API documentation is available via Swagger at `http://localhost:5000/