from typing import Dict, Any

def calculate_score(analysis_result: Dict[str, Any], plagiarism_report: Dict[str, Any]) -> float:
    complexity_score = min(analysis_result['complexity'] / 10, 1) * 40
    originality_score = (1 - plagiarism_report['overall_similarity']) * 60
    return complexity_score + originality_score
