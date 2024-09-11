import os
from dotenv import load_dotenv
load_dotenv()
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = 'project_verification'
COLLECTION_NAME = 'projects'