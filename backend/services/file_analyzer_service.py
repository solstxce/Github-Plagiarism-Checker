import radon.complexity as rc
from radon.metrics import h_visit
from radon.raw import analyze
from typing import Dict, Any

class FileAnalyzerService:
    def analyze_files(self, files_content: Dict[str, str]) -> Dict[str, Any]:
        total_complexity = 0
        total_lines = 0
        halstead_metrics = {"h1": 0, "h2": 0, "N1": 0, "N2": 0}
        
        for filename, content in files_content.items():
            if filename.endswith('.py'):
                complexity = rc.cc_visit(content)
                total_complexity += sum(item.complexity for item in complexity)
            
            raw_metrics = analyze(content)
            total_lines += raw_metrics.loc
            
            h_metrics = h_visit(content)
            for key in halstead_metrics:
                halstead_metrics[key] += getattr(h_metrics, key)
        
        avg_complexity = total_complexity / len(files_content) if files_content else 0
        
        return {
            "complexity": avg_complexity,
            "total_lines": total_lines,
            "halstead_metrics": halstead_metrics
        }