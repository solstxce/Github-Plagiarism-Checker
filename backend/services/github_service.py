import requests
import base64
from ratelimit import limits, sleep_and_retry
from typing import Dict

class GitHubService:
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"token {self.token}"})

    @sleep_and_retry
    @limits(calls=30, period=60)
    def get_repository_files(self, owner: str, repo: str, path: str = "") -> Dict[str, str]:
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        response = self.session.get(url)
        response.raise_for_status()
        contents = response.json()
        
        files_content = {}
        for item in contents:
            if item["type"] == "file":
                if item["name"].endswith((".py", ".html", ".css", ".js", ".java", ".jsp")):
                    file_content = self._get_file_content(item["download_url"])
                    files_content[item["name"]] = file_content
            elif item["type"] == "dir":
                files_content.update(self.get_repository_files(owner, repo, item["path"]))
        
        return files_content

    def _get_file_content(self, file_url: str) -> str:
        response = self.session.get(file_url)
        response.raise_for_status()
        content = base64.b64decode(response.json()["content"]).decode('utf-8')
        return content
