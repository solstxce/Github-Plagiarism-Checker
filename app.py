import streamlit as st
import pandas as pd
import requests
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
from ratelimit import limits, sleep_and_retry
from dotenv import load_dotenv
load_dotenv()
import os
import logging
logging.basicConfig(level=logging.INFO)
import re

import logging
import ast
import math


# Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
DB_NAME = 'project_verification'
COLLECTION_NAME = 'projects'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', 'YOUR_GITHUB_TOKEN')

# Models
@dataclass
class Project:
    roll_no: str
    email: str  # New field
    github_link: str
    complexity: float
    total_lines: int
    halstead_metrics: Dict[str, float]
    files_content: Dict[str, str]
    score: float = 0.0
    plagiarism_report: Dict[str, Any] = field(default_factory=dict)

# Services
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


class FileAnalyzerService:
    # def analyze_files(self, files_content: Dict[str, str]) -> Dict[str, Any]:
    #     total_complexity = 0
    #     total_lines = 0
    #     total_halstead_difficulty = 0
        
    #     for filename, content in files_content.items():
    #         file_complexity = self.calculate_cyclomatic_complexity(content)
    #         file_lines = len(content.split('\n'))
    #         file_halstead = self.calculate_halstead_metrics(content)
            
    #         total_complexity += file_complexity
    #         total_lines += file_lines
    #         total_halstead_difficulty += file_halstead['difficulty']
        
    #     num_files = len(files_content)
    #     avg_complexity = total_complexity / num_files if num_files else 0
    #     avg_halstead_difficulty = total_halstead_difficulty / num_files if num_files else 0
        
    #     # Normalize complexity to a 0-100 scale
    #     normalized_complexity = min(100, (avg_complexity / 10) * 50 + (avg_halstead_difficulty / 30) * 50)
        
    #     return {
    #         "complexity": normalized_complexity,
    #         "total_lines": total_lines,
    #         "avg_cyclomatic_complexity": avg_complexity,
    #         "avg_halstead_difficulty": avg_halstead_difficulty
    #     }

    def analyze_files(self, files_content: Dict[str, str]) -> Dict[str, Any]:
        total_complexity = 0
        total_lines = 0
        total_halstead_difficulty = 0
        
        for filename, content in files_content.items():
            file_complexity = self.calculate_cyclomatic_complexity(content)
            file_lines = len(content.split('\n'))
            file_halstead = self.calculate_halstead_metrics(content)
            
            total_complexity += file_complexity
            total_lines += file_lines
            total_halstead_difficulty += file_halstead['difficulty']
        
        num_files = len(files_content)
        avg_complexity = total_complexity / num_files if num_files else 0
        avg_halstead_difficulty = total_halstead_difficulty / num_files if num_files else 0
        
        # Normalize complexity to a 0-100 scale
        normalized_complexity = self.normalize_complexity(avg_complexity, avg_halstead_difficulty)
        
        return {
            "complexity": normalized_complexity,
            "total_lines": total_lines,
            "avg_cyclomatic_complexity": avg_complexity,
            "avg_halstead_difficulty": avg_halstead_difficulty
        }

    def normalize_complexity(self, cyclomatic_complexity: float, halstead_difficulty: float) -> float:
        # Normalize cyclomatic complexity (typical range 1-10, but can be higher)
        norm_cyclomatic = min(100, (cyclomatic_complexity / 10) * 100)
        
        # Normalize Halstead difficulty (typical range 0-100, but can be higher)
        norm_halstead = min(100, (halstead_difficulty / 50) * 100)
        
        # Combine the two metrics (equal weight)
        return (norm_cyclomatic + norm_halstead) / 2

    def calculate_cyclomatic_complexity(self, code: str) -> int:
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1

            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)

            def visit_FunctionDef(self, node):
                self.complexity += 1
                self.generic_visit(node)

        try:
            tree = ast.parse(code)
            visitor = ComplexityVisitor()
            visitor.visit(tree)
            return visitor.complexity
        except SyntaxError:
            # If we can't parse the code (e.g., it's not Python), return a default value
            return 1

    def calculate_halstead_metrics(self, code: str) -> Dict[str, float]:
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.operator) or isinstance(node, ast.Compare):
                    operators.add(type(node).__name__)
                    total_operators += 1
                elif isinstance(node, ast.Name):
                    operands.add(node.id)
                    total_operands += 1

            n1 = len(operators)
            n2 = len(operands)
            N1 = total_operators
            N2 = total_operands

            program_length = N1 + N2
            vocabulary = n1 + n2
            volume = program_length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
            effort = difficulty * volume

            return {
                "vocabulary": vocabulary,
                "length": program_length,
                "volume": volume,
                "difficulty": difficulty,
                "effort": effort
            }
        except SyntaxError:
            # If we can't parse the code, return default values
            return {
                "vocabulary": 0,
                "length": 0,
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            }

# class FileAnalyzerService:
#     def analyze_files(self, files_content: Dict[str, str]) -> Dict[str, Any]:
#         total_complexity = 0
#         total_lines = 0
#         halstead_metrics = {"h1": 0, "h2": 0, "N1": 0, "N2": 0}
        
#         for filename, content in files_content.items():
#             # Placeholder for complexity calculation
#             total_complexity += len(content.split('\n'))
#             total_lines += len(content.split('\n'))
            
#             # Placeholder for Halstead metrics calculation
#             for key in halstead_metrics:
#                 halstead_metrics[key] += len(content)
        
#         avg_complexity = total_complexity / len(files_content) if files_content else 0
        
#         return {
#             "complexity": avg_complexity,
#             "total_lines": total_lines,
#             "halstead_metrics": halstead_metrics
#         }

class GitHubService:
    BASE_URL = "https://api.github.com"
    RAW_CONTENT_BASE_URL = "https://raw.githubusercontent.com"
    
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        })

    @sleep_and_retry
    @limits(calls=30, period=60)
    def get_repository_files(self, owner: str, repo: str, path: str = "") -> Dict[str, str]:
        """
        Recursively fetch all files from a GitHub repository.
        
        :param owner: The owner of the repository
        :param repo: The name of the repository
        :param path: The path within the repository to fetch (used for recursion)
        :return: A dictionary mapping file paths to their contents
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{path}"
        logging.info(f"Fetching contents from GitHub: {url}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            contents = response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch from GitHub: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from GitHub response: {str(e)}")
            raise

        files_content = {}
        for item in contents:
            if item["type"] == "file":
                if self._is_valid_file_type(item["name"]):
                    file_content = self._get_file_content(owner, repo, item["path"])
                    if file_content:
                        files_content[item["path"]] = file_content
                    logging.info(f"Fetched file: {item['path']}")
            elif item["type"] == "dir":
                logging.info(f"Recursing into directory: {item['path']}")
                files_content.update(self.get_repository_files(owner, repo, item["path"]))
        
        return files_content

    def _is_valid_file_type(self, filename: str) -> bool:
        """
        Check if the file type is one we want to analyze.
        
        :param filename: The name of the file
        :return: True if the file type is valid, False otherwise
        """
        valid_extensions = (".py", ".js", ".java", ".html", ".css", ".cpp", ".c", ".h",".jsp",".md")
        return filename.lower().endswith(valid_extensions)

    def _get_file_content(self, owner: str, repo: str, file_path: str) -> str:
        """
        Fetch the content of a single file from GitHub, trying both API and raw content URL.
        
        :param owner: The owner of the repository
        :param repo: The name of the repository
        :param file_path: The path of the file within the repository
        :return: The content of the file as a string, or None if both methods fail
        """
        logging.info(f"Fetching file content: {file_path}")
        
        # Try GitHub API first
        try:
            return self._get_file_content_api(owner, repo, file_path)
        except Exception as e:
            logging.warning(f"Failed to fetch file content via API: {str(e)}")
        
        # If API fails, try raw content URL
        try:
            return self._get_file_content_raw(owner, repo, file_path)
        except Exception as e:
            logging.error(f"Failed to fetch file content via raw URL: {str(e)}")
        
        return None

    def _get_file_content_api(self, owner: str, repo: str, file_path: str) -> str:
        """Fetch file content using GitHub API"""
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/contents/{file_path}"
        response = self.session.get(url)
        response.raise_for_status()
        content = response.json()["content"]
        return base64.b64decode(content).decode('utf-8')

    def _get_file_content_raw(self, owner: str, repo: str, file_path: str) -> str:
        """Fetch file content using raw.githubusercontent.com"""
        url = f"{self.RAW_CONTENT_BASE_URL}/{owner}/{repo}/main/{file_path}"
        response = requests.get(url)
        response.raise_for_status()
        return response.text

    def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Fetch general information about the repository.
        
        :param owner: The owner of the repository
        :param repo: The name of the repository
        :return: A dictionary containing repository information
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}"
        logging.info(f"Fetching repository info from GitHub: {url}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch repository info: {str(e)}")
            raise

    def get_commit_history(self, owner: str, repo: str, max_commits: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch the commit history of the repository.
        
        :param owner: The owner of the repository
        :param repo: The name of the repository
        :param max_commits: Maximum number of commits to fetch
        :return: A list of dictionaries containing commit information
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/commits"
        params = {"per_page": min(max_commits, 100)}  # GitHub's max per page is 100
        logging.info(f"Fetching commit history from GitHub: {url}")
        
        commits = []
        try:
            while len(commits) < max_commits:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                page_commits = response.json()
                if not page_commits:
                    break
                commits.extend(page_commits)
                if 'next' not in response.links:
                    break
                url = response.links['next']['url']
            return commits[:max_commits]
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch commit history: {str(e)}")
            raise

    def get_languages(self, owner: str, repo: str) -> Dict[str, int]:
        """
        Fetch the languages used in the repository and their byte count.
        
        :param owner: The owner of the repository
        :param repo: The name of the repository
        :return: A dictionary mapping language names to byte counts
        """
        url = f"{self.BASE_URL}/repos/{owner}/{repo}/languages"
        logging.info(f"Fetching language information from GitHub: {url}")
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch language information: {str(e)}")
            raise

# class PlagiarismCheckerService:
#     def check_plagiarism(self, current_project: Dict[str, Any], existing_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
#         vectorizer = TfidfVectorizer()
#         overall_similarity = 0
#         file_similarities = {}
#         most_similar_projects = {}
#         plagiarism_sources = {}

#         for filename, content in current_project['files_content'].items():
#             preprocessed_current = self.preprocess_code(content)
#             existing_contents = [self.preprocess_code(project['files_content'].get(filename, ''))
#                                  for project in existing_projects]

#             if not existing_contents:
#                 file_similarities[filename] = 0
#                 continue

#             all_contents = [preprocessed_current] + existing_contents
#             tfidf_matrix = vectorizer.fit_transform(all_contents)

#             cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
#             max_similarity = np.max(cosine_similarities)
#             most_similar_index = np.argmax(cosine_similarities)

#             file_similarities[filename] = max_similarity * 100
#             if max_similarity > 0:
#                 most_similar_projects[filename] = existing_projects[most_similar_index]['roll_no']
                
#                 plagiarism_sources[filename] = {
#                     'roll_no': existing_projects[most_similar_index]['roll_no'],
#                     'similarity': max_similarity * 100,
#                     'github_link': existing_projects[most_similar_index]['github_link']
#                 }

#             overall_similarity += max_similarity

#         overall_similarity /= len(current_project['files_content']) if current_project['files_content'] else 1

#         return {
#             "overall_similarity": overall_similarity,
#             "file_similarities": file_similarities,
#             "most_similar_projects": most_similar_projects,
#             "plagiarism_sources": plagiarism_sources
#         }

#     def preprocess_code(self, code: str) -> str:
#         # Placeholder for code preprocessing
#         return code.lower()

# class PlagiarismCheckerService:
#     def check_plagiarism(self, current_project: Dict[str, Any], existing_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
#         vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
#         overall_similarity = 0
#         file_similarities = {}
#         most_similar_projects = {}
#         plagiarism_sources = {}

#         for filename, content in current_project['files_content'].items():
#             preprocessed_current = self.preprocess_code(content)
#             existing_contents = [self.preprocess_code(project['files_content'].get(filename, ''))
#                                  for project in existing_projects]

#             if not existing_contents:
#                 file_similarities[filename] = 0
#                 continue

#             all_contents = [preprocessed_current] + existing_contents
#             tfidf_matrix = vectorizer.fit_transform(all_contents)

#             cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
#             max_similarity = np.max(cosine_similarities)
#             most_similar_index = np.argmax(cosine_similarities)

#             file_similarities[filename] = max_similarity * 100
#             if max_similarity > 0:
#                 most_similar_projects[filename] = existing_projects[most_similar_index]['roll_no']
                
#                 plagiarism_sources[filename] = {
#                     'roll_no': existing_projects[most_similar_index]['roll_no'],
#                     'similarity': max_similarity * 100,
#                     'github_link': existing_projects[most_similar_index]['github_link']
#                 }

#             overall_similarity += max_similarity

#         overall_similarity = (overall_similarity / len(current_project['files_content'])) * 100 if current_project['files_content'] else 0

#         return {
#             "overall_similarity": overall_similarity,
#             "file_similarities": file_similarities,
#             "most_similar_projects": most_similar_projects,
#             "plagiarism_sources": plagiarism_sources
#         }

#     def preprocess_code(self, code: str) -> str:
#         # Remove comments
#         code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.S)
#         # Remove whitespace
#         code = re.sub(r'\s+', '', code)
#         # Convert to lowercase
#         return code.lower()


class PlagiarismCheckerService:
    def check_plagiarism(self, current_project: Dict[str, Any], existing_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
        overall_similarity = 0
        file_similarities = {}
        most_similar_projects = {}
        plagiarism_sources = {}

        for current_filename, current_content in current_project['files_content'].items():
            current_ext = os.path.splitext(current_filename)[1].lower()
            preprocessed_current = self.preprocess_code(current_content)
            
            similar_files = []
            for project in existing_projects:
                for filename, content in project['files_content'].items():
                    if os.path.splitext(filename)[1].lower() == current_ext:
                        similar_files.append({
                            'roll_no': project['roll_no'],
                            'github_link': project['github_link'],
                            'content': self.preprocess_code(content)
                        })

            if not similar_files:
                file_similarities[current_filename] = 0
                continue

            all_contents = [preprocessed_current] + [file['content'] for file in similar_files]
            tfidf_matrix = vectorizer.fit_transform(all_contents)

            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            max_similarity = np.max(cosine_similarities)
            most_similar_index = np.argmax(cosine_similarities)

            file_similarities[current_filename] = max_similarity * 100
            if max_similarity > 0:
                most_similar_projects[current_filename] = similar_files[most_similar_index]['roll_no']
                
                plagiarism_sources[current_filename] = {
                    'roll_no': similar_files[most_similar_index]['roll_no'],
                    'similarity': max_similarity * 100,
                    'github_link': similar_files[most_similar_index]['github_link']
                }

            overall_similarity += max_similarity

        overall_similarity = (overall_similarity / len(current_project['files_content'])) * 100 if current_project['files_content'] else 0

        return {
            "overall_similarity": overall_similarity,
            "file_similarities": file_similarities,
            "most_similar_projects": most_similar_projects,
            "plagiarism_sources": plagiarism_sources
        }

    def preprocess_code(self, code: str) -> str:
        # Remove comments
        code = re.sub(r'//.*?\n|/\*.*?\*/', '', code, flags=re.S)
        # Remove whitespace
        code = re.sub(r'\s+', '', code)
        # Convert to lowercase
        return code.lower()


def calculate_score(analysis_result: Dict[str, Any], plagiarism_report: Dict[str, Any]) -> Dict[str, float]:
    try:
        # Normalize complexity score (0-100 scale)
        complexity_score = min(analysis_result['complexity'], 100)
        
        # Calculate originality score (100 - similarity)
        originality_score = 100 - plagiarism_report['overall_similarity']
        
        # Calculate final score
        # Adjust weights: 40% complexity, 60% originality
        final_score = (0.08 * complexity_score) + (0.92 * originality_score)
        
        logging.info(f"Score calculation: complexity_score={complexity_score:.2f}, originality_score={originality_score:.2f}, final_score={final_score:.2f}")
        
        return {
            "complexity_score": round(complexity_score, 2),
            "originality_score": round(originality_score, 2),
            "final_score": round(final_score, 2)
        }
    except Exception as e:
        logging.error(f"Error in calculate_score: {str(e)}", exc_info=True)
        return {
            "complexity_score": 0,
            "originality_score": 0,
            "final_score": 0
        }
# Utility functions
# def calculate_score(analysis_result: Dict[str, Any], plagiarism_report: Dict[str, Any]) -> float:
#     complexity_score = min(analysis_result['complexity'] / 10, 1) * 40
#     originality_score = (1 - plagiarism_report['overall_similarity']) * 60
#     return complexity_score + originality_score

# def calculate_score(analysis_result: Dict[str, Any], plagiarism_report: Dict[str, Any]) -> float:
#     try:
#         # Normalize complexity to a 0-100 scale
#         complexity_score = min(analysis_result['complexity'], 100)
        
#         # Invert the plagiarism score (higher similarity = lower score)
#         originality_score = 100 - plagiarism_report['overall_similarity']
        
#         # Weighted average: 40% complexity, 60% originality
#         score = (0.2 * complexity_score) + (0.8 * originality_score)
        
#         logging.info(f"Score calculation: complexity_score={complexity_score:.2f}, originality_score={originality_score:.2f}")
#         return round(score, 2)
#     except Exception as e:
#         logging.error(f"Error in calculate_score: {str(e)}", exc_info=True)
#         return 0.0


def generate_report(project: Project) -> str:
    report = f"Project Report for Roll No: {project.roll_no}\n"
    report += f"Email: {project.email}\n"
    report += f"Complexity: {project.complexity:.2f}\n"
    report += f"Originality: {100 - project.plagiarism_report['overall_similarity']:.2f}\n"
    report += f"Final Score: {project.score:.2f}/100\n\n"
    
    report += "Plagiarism Report:\n"
    for file, similarity in project.plagiarism_report['file_similarities'].items():
        report += f"  {file}: {similarity:.2f}% similar\n"
        if similarity > 30:
            source = project.plagiarism_report['plagiarism_sources'][file]
            report += f"    Potential source: Roll No. {source['roll_no']} (Similarity: {source['similarity']:.2f}%)\n"
            report += f"    GitHub Link: {source['github_link']}\n"
    
    if project.plagiarism_report['overall_similarity'] > 30:
        report += "\nWarning: High similarity detected. Please review the code for potential plagiarism.\n"
    
    return report

# Initialize services
github_service = GitHubService(GITHUB_TOKEN)
file_analyzer_service = FileAnalyzerService()
plagiarism_checker_service = PlagiarismCheckerService()
db_service = DBService()



# Main verification function

import logging
from typing import Dict, Any
from github import Github, GithubException

# def verify_project(roll_no: str, github_link: str) -> Dict[str, Any]:
#     logging.info(f"Starting verification for roll_no: {roll_no}, github_link: {github_link}")
    
#     result = {
#         "roll_no": roll_no,
#         "score": None,
#         "complexity": None,
#         "originality": None,
#         "plagiarism_report": None,
#         "report": None,
#         "error": None
#     }
    
#     try:
#         # Extract owner and repo from github_link
#         try:
#             _, _, _, owner, repo = github_link.rstrip('/').split('/')
#             logging.info(f"Extracted owner: {owner}, repo: {repo}")
#         except ValueError:
#             raise ValueError(f"Invalid GitHub link format: {github_link}")
        
#         # Verify GitHub access
#         try:
#             g = Github(GITHUB_TOKEN)
#             repository = g.get_repo(f"{owner}/{repo}")
#             logging.info(f"Successfully accessed GitHub repository: {repository.full_name}")
#         except GithubException as e:
#             logging.error(f"Failed to access GitHub repository: {str(e)}")
#             raise
        
#         # Fetch and analyze files
#         try:
#             files_content = github_service.get_repository_files(owner, repo)
#             logging.info(f"Fetched {len(files_content)} files from GitHub")
#             if not files_content:
#                 raise ValueError("No files fetched from the repository")
#         except Exception as e:
#             logging.error(f"Failed to fetch files from GitHub: {str(e)}")
#             raise
        
#         try:

#             analysis_result = file_analyzer_service.analyze_files(files_content)
#             logging.info(f"File analysis result: {analysis_result}")
            
#             existing_projects = db_service.get_all_projects()
#             plagiarism_report = plagiarism_checker_service.check_plagiarism(
#                 {"files_content": files_content}, existing_projects
#             )
#             logging.info(f"Plagiarism check completed: {plagiarism_report['overall_similarity']:.2f} overall similarity")
            
#             scores = calculate_score(analysis_result, plagiarism_report)
#             logging.info(f"Calculated scores: {scores}")
            
#             # Create project object
#             project = Project(
#                 roll_no=roll_no,
#                 github_link=github_link,
#                 complexity=analysis_result['complexity'],
#                 total_lines=analysis_result['total_lines'],
#                 halstead_metrics=analysis_result.get('halstead_metrics', {}),
#                 files_content=files_content,
#                 score=scores['final_score'],
#                 plagiarism_report=plagiarism_report
#             )
            
#             # Save project data
#             db_service.save_project(project)
#             logging.info("Project data saved to database")
            
#             # Generate report
#             report = generate_report(project)
            
#             result.update({
#                 "score": scores['final_score'],
#                 "complexity_score": scores['complexity_score'],
#                 "originality_score": scores['originality_score'],
#                 "complexity": analysis_result['complexity'],
#                 "plagiarism_report": plagiarism_report,
#                 "report": report
#             })
            
#         except Exception as e:
#             logging.error(f"Error in verify_project: {str(e)}", exc_info=True)
#             result["error"] = f"Failed to verify project: {str(e)}"
    
#         return result


def verify_project(email: str, roll_no: str, github_link: str) -> Dict[str, Any]:
    logging.info(f"Starting verification for email: {email}, roll_no: {roll_no}, github_link: {github_link}")
    
    result = {
        "email": email,
        "roll_no": roll_no,
        "score": None,
        "complexity": None,
        "originality": None,
        "plagiarism_report": None,
        "report": None,
        "error": None
    }
   
    try:
        # Extract owner and repo from github_link
        try:
            _, _, _, owner, repo = github_link.rstrip('/').split('/')
            logging.info(f"Extracted owner: {owner}, repo: {repo}")
        except ValueError:
            raise ValueError(f"Invalid GitHub link format: {github_link}")
       
        # Verify GitHub access
        try:
            g = Github(GITHUB_TOKEN)
            repository = g.get_repo(f"{owner}/{repo}")
            logging.info(f"Successfully accessed GitHub repository: {repository.full_name}")
        except GithubException as e:
            logging.error(f"Failed to access GitHub repository: {str(e)}")
            raise
       
        # Fetch and analyze files
        try:
            files_content = github_service.get_repository_files(owner, repo)
            logging.info(f"Fetched {len(files_content)} files from GitHub")
            if not files_content:
                raise ValueError("No files fetched from the repository")
        except Exception as e:
            logging.error(f"Failed to fetch files from GitHub: {str(e)}")
            raise
       
        try:
            analysis_result = file_analyzer_service.analyze_files(files_content)
            logging.info(f"File analysis result: {analysis_result}")
           
            existing_projects = db_service.get_all_projects()
            plagiarism_report = plagiarism_checker_service.check_plagiarism(
                {"files_content": files_content}, existing_projects
            )
            logging.info(f"Plagiarism check completed: {plagiarism_report['overall_similarity']:.2f} overall similarity")
           
            scores = calculate_score(analysis_result, plagiarism_report)
            logging.info(f"Calculated scores: {scores}")
           
            # Create project object
            project = Project(
                email=email,
                roll_no=roll_no,
                github_link=github_link,
                complexity=analysis_result['complexity'],
                total_lines=analysis_result['total_lines'],
                halstead_metrics=analysis_result.get('halstead_metrics', {}),
                files_content=files_content,
                score=scores['final_score'],
                plagiarism_report=plagiarism_report
            )
           
            # Save project data
            db_service.save_project(project)
            logging.info("Project data saved to database")
           
            # Generate report
            report = generate_report(project)
           
            result.update({
                "score": scores['final_score'],
                "complexity_score": scores['complexity_score'],
                "originality_score": scores['originality_score'],
                "complexity": analysis_result['complexity'],
                "plagiarism_report": plagiarism_report,
                "report": report
            })
           
        except Exception as e:
            logging.error(f"Error in verify_project: {str(e)}", exc_info=True)
            result["error"] = f"Failed to verify project: {str(e)}"
   
    except Exception as e:
        logging.error(f"Unexpected error in verify_project: {str(e)}", exc_info=True)
        result["error"] = f"Unexpected error: {str(e)}"
   
    return result


# Streamlit UI
def process_file(uploaded_file):
    df = pd.read_csv(uploaded_file)
    results = []
    for _, row in df.iterrows():
        result = verify_project(row["email"], row["rollno"], row["github"])
        results.append(result)
    return results

def render_file_uploader():
    return st.file_uploader("Upload CSV file", type="csv")


def render_results(results):
    if not results:
        st.write("No results to display.")
        return

    st.write("Overall Results:")
    
    # Create a list of dictionaries with only the essential information
    simplified_results = []
    for result in results:
        simplified_result = {
            'Roll No': result.get('roll_no', 'Unknown'),
            'Score': result.get('score', 'N/A'),
            'Complexity Score': result.get('complexity_score', 'N/A'),
            'Originality Score': result.get('originality_score', 'N/A')
        }
        simplified_results.append(simplified_result)

    # Display the simplified results as a dataframe
    df = pd.DataFrame(simplified_results)
    st.dataframe(df)

    # Create a download link for the CSV
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="verification_results.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.subheader("Detailed Analysis")
    for result in results:
        with st.expander(f"Detailed Report for Roll No: {result.get('roll_no', 'Unknown')}"):
            if 'error' in result:
                st.error(f"Error: {result['error']}")
                continue
            
            st.write(f"Final Score: {result.get('score', 'N/A')}")
            st.write(f"Complexity Score: {result.get('complexity_score', 'N/A')}")
            st.write(f"Originality Score: {result.get('originality_score', 'N/A')}")
            
            if 'plagiarism_report' in result:
                st.subheader("Plagiarism Report")
                overall_similarity = result['plagiarism_report'].get('overall_similarity', 'N/A')
                st.write(f"Overall Similarity: {overall_similarity}")
                
                file_similarities = result['plagiarism_report'].get('file_similarities', {})
                if file_similarities:
                    st.write("File Similarities:")
                    for file, similarity in file_similarities.items():
                        st.write(f"- {file}: {similarity}% similar")

            if 'report' in result:
                st.text_area("Full Report", result['report'], height=200)

    st.success("Analysis complete. Expand the detailed reports for more information.")
# def render_results(results):
#     if not results:
#         st.write("No results to display.")
#         return

#     df = pd.DataFrame(results)
#     st.write("Overall Results:")
    
#     display_columns = ['roll_no', 'score', 'complexity_score', 'originality_score']
#     existing_columns = [col for col in display_columns if col in df.columns]
    
#     if existing_columns:
#         st.dataframe(df[existing_columns])
#     else:
#         st.write("No valid columns to display.")

#     # Create a download link for the CSV
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="verification_results.csv">Download CSV File</a>'
#     st.markdown(href, unsafe_allow_html=True)

#     st.subheader("Detailed Analysis for Concerning Cases")
#     for result in results:
#         # Only show detailed analysis for low scores or high similarity
#         if result.get('score', 100) < 50 or result.get('originality_score', 100) < 70:
#             with st.expander(f"Detailed Report for Roll No: {result.get('roll_no', 'Unknown')}"):
#                 if 'error' in result:
#                     st.error(f"Error: {result['error']}")
#                     continue
                
#                 st.write(f"Final Score: {result.get('score', 'N/A')}")
#                 st.write(f"Complexity Score: {result.get('complexity_score', 'N/A')}")
#                 st.write(f"Originality Score: {result.get('originality_score', 'N/A')}")
                
#                 # ... (rest of the function remains the same)

# def render_results(results):
#     if not results:
#         st.write("No results to display.")
#         return

#     df = pd.DataFrame(results)
#     st.write("Overall Results:")
    
#     # Define the columns we want to display if they exist
#     display_columns = ['roll_no', 'score', 'complexity', 'originality']
    
#     # Filter the columns to only include those that exist in the DataFrame
#     existing_columns = [col for col in display_columns if col in df.columns]
    
#     if existing_columns:
#         st.dataframe(df[existing_columns])
#     else:
#         st.write("No valid columns to display.")

#     # Create a download link for the CSV
#     csv = df.to_csv(index=False)
#     b64 = base64.b64encode(csv.encode()).decode()
#     href = f'<a href="data:file/csv;base64,{b64}" download="verification_results.csv">Download CSV File</a>'
#     st.markdown(href, unsafe_allow_html=True)

#     st.subheader("Detailed Analysis")
#     for result in results:
#         with st.expander(f"Detailed Report for Roll No: {result.get('roll_no', 'Unknown')}"):
#             if 'error' in result:
#                 st.error(f"Error: {result['error']}")
#                 continue
            
#             st.write(f"Score: {result.get('score', 'N/A')}")
#             st.write(f"Complexity: {result.get('complexity', 'N/A')}")
#             st.write(f"Originality: {result.get('originality', 'N/A')}")
            
#             if 'plagiarism_report' in result:
#                 st.subheader("Plagiarism Report")
#                 for file, similarity in result['plagiarism_report'].get('file_similarities', {}).items():
#                     st.write(f"{file}: {similarity:.2f}% similar")
#                     if similarity > 30 and 'plagiarism_sources' in result['plagiarism_report']:
#                         source = result['plagiarism_report']['plagiarism_sources'].get(file, {})
#                         if source:
#                             st.write(f"  Potential source: Roll No. {source.get('roll_no', 'Unknown')} (Similarity: {source.get('similarity', 'N/A')}%)")
#                             st.write(f"  GitHub Link: {source.get('github_link', 'N/A')}")
                
#                 if result['plagiarism_report'].get('overall_similarity', 0) > 0.3:
#                     st.warning("High similarity detected. Please review the code for potential plagiarism.")

#             if 'report' in result:
#                 st.text(result['report'])

def main():
    st.set_page_config(page_title="Project Verification System", layout="wide")
    st.title("Project Verification System")

    uploaded_file = render_file_uploader()

    if uploaded_file is not None:
        with st.spinner("Processing file..."):
            results = process_file(uploaded_file)
        render_results(results)

if __name__ == "__main__":
    main()