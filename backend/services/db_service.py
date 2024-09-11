from pymongo import MongoClient
from typing import List, Dict, Any
from models.project import Project
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME

class DBService:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client[DB_NAME]
        self.projects = self.db[COLLECTION_NAME]

    def save_project(self, project: Project):
        project_data = project.__dict__
        self.projects.update_one({"roll_no": project.roll_no}, {"$set": project_data}, upsert=True)

    def get_all_projects(self) -> List[Dict[str, Any]]:
        return list(self.projects.find({}, {"_id": 0}))