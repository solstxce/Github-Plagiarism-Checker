from .github_service import GitHubService
from .file_analyzer_service import FileAnalyzerService
from .plagiarism_checker_service import PlagiarismCheckerService
from .db_service import DBService
from utils.scoring import calculate_score
from models.project import Project
from config import GITHUB_TOKEN

github_service = GitHubService(GITHUB_TOKEN)
file_analyzer_service = FileAnalyzerService()
plagiarism_checker_service = PlagiarismCheckerService()
db_service = DBService()

def verify_project(roll_no: str, github_link: str) -> dict:
    try:
        # Extract owner and repo from github_link
        _, _, _, owner, repo = github_link.rstrip('/').split('/')
        
        # Fetch and analyze files
        files_content = github_service.get_repository_files(owner, repo)
        analysis_result = file_analyzer_service.analyze_files(files_content)
        
        # Check for plagiarism
        existing_projects = db_service.get_all_projects()
        plagiarism_report = plagiarism_checker_service.check_plagiarism(analysis_result, existing_projects)
        
        # Calculate score
        score = calculate_score(analysis_result, plagiarism_report)
        
        # Create project object
        project = Project(
            roll_no=roll_no,
            github_link=github_link,
            complexity=analysis_result['complexity'],
            total_lines=analysis_result['total_lines'],
            halstead_metrics=analysis_result['halstead_metrics'],
            files_content=files_content,
            score=score,
            plagiarism_report=plagiarism_report
        )
        
        # Save project data
        db_service.save_project(project)
        
        # Generate report
        report = generate_report(project)
        
        return {
            "roll_no": roll_no,
            "score": score,
            "complexity": analysis_result['complexity'],
            "originality": 1 - plagiarism_report['overall_similarity'],
            "plagiarism_report": plagiarism_report,
            "report": report
        }
    except Exception as e:
        return {
            "roll_no": roll_no,
            "error": f"Failed to verify project: {str(e)}"
        }

def generate_report(project: Project) -> str:
    report = f"Project Report for Roll No: {project.roll_no}\n"
    report += f"Complexity: {project.complexity:.2f}\n"
    report += f"Originality: {1 - project.plagiarism_report['overall_similarity']:.2f}\n"
    report += f"Final Score: {project.score:.2f}/100\n\n"
    
    report += "Plagiarism Report:\n"
    for file, similarity in project.plagiarism_report['file_similarities'].items():
        report += f"  {file}: {similarity:.2f}% similar\n"
    
    if project.plagiarism_report['overall_similarity'] > 0.3:
        report += "\nWarning: High similarity detected. Please review the code for potential plagiarism.\n"
    
    return report
