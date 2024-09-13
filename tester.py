import requests
import base64
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# GitHub API configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_API_BASE_URL = "https://api.github.com"
RAW_CONTENT_BASE_URL = "https://raw.githubusercontent.com"

# Repository details (you can change these)
OWNER = "demogitpace"
REPO = "project-23"

class GitHubRepoTester:
    def __init__(self, owner: str, repo: str, token: str):
        self.owner = owner
        self.repo = repo
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        })

    def get_repo_contents(self, path: str = "") -> List[Dict[str, Any]]:
        url = f"{GITHUB_API_BASE_URL}/repos/{self.owner}/{self.repo}/contents/{path}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_file_content_api(self, file_path: str) -> str:
        url = f"{GITHUB_API_BASE_URL}/repos/{self.owner}/{self.repo}/contents/{file_path}"
        response = self.session.get(url)
        response.raise_for_status()
        content = response.json()["content"]
        return base64.b64decode(content).decode('utf-8')

    def get_file_content_raw(self, file_path: str) -> str:
        url = f"{RAW_CONTENT_BASE_URL}/{self.owner}/{self.repo}/main/{file_path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def test_repo(self):
        def process_contents(contents: List[Dict[str, Any]], path: str = ""):
            for item in contents:
                if item["type"] == "file":
                    self.test_file(item["path"])
                elif item["type"] == "dir":
                    new_contents = self.get_repo_contents(item["path"])
                    process_contents(new_contents, item["path"])

        initial_contents = self.get_repo_contents()
        process_contents(initial_contents)

    def test_file(self, file_path: str):
        print(f"\nTesting file: {file_path}")
        
        # Test GitHub API
        print("Using GitHub API:")
        try:
            content_api = self.get_file_content_api(file_path)
            print(f"  Success! Content length: {len(content_api)}")
            print(f"  First 100 characters: {content_api[:100]}")
        except Exception as e:
            print(f"  Error: {str(e)}")

        # Test raw.githubusercontent.com
        print("Using raw.githubusercontent.com:")
        try:
            content_raw = self.get_file_content_raw(file_path)
            print(f"  Success! Content length: {len(content_raw)}")
            print(f"  First 100 characters: {content_raw[:100]}")
        except Exception as e:
            print(f"  Error: {str(e)}")

        # Compare results
        if 'content_api' in locals() and 'content_raw' in locals():
            if content_api == content_raw:
                print("  Results match!")
            else:
                print("  Results do not match.")
                print(f"  API content length: {len(content_api)}")
                print(f"  Raw content length: {len(content_raw)}")
        else:
            print("  Unable to compare results due to errors.")

def main():
    tester = GitHubRepoTester(OWNER, REPO, GITHUB_TOKEN)
    try:
        tester.test_repo()
    except Exception as e:
        print(f"An error occurred while testing the repository: {str(e)}")

if __name__ == "__main__":
    main()