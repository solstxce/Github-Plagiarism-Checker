from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Any
from utils.preprocessing import preprocess_code

class PlagiarismCheckerService:
    def check_plagiarism(self, current_project: Dict[str, Any], existing_projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        vectorizer = TfidfVectorizer()
        overall_similarity = 0
        file_similarities = {}
        most_similar_projects = {}
        
        for filename, content in current_project['files_content'].items():
            preprocessed_current = preprocess_code(content)
            existing_contents = [preprocess_code(project['files_content'].get(filename, '')) 
                                 for project in existing_projects]
            
            if not existing_contents:
                file_similarities[filename] = 0
                continue
            
            all_contents = [preprocessed_current] + existing_contents
            tfidf_matrix = vectorizer.fit_transform(all_contents)
            
            cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            max_similarity = np.max(cosine_similarities)
            most_similar_index = np.argmax(cosine_similarities)
            
            file_similarities[filename] = max_similarity * 100
            if max_similarity > 0:
                most_similar_projects[filename] = existing_projects[most_similar_index]['roll_no']
            
            overall_similarity += max_similarity
        
        overall_similarity /= len(current_project['files_content']) if current_project['files_content'] else 1
        
        return {
            "overall_similarity": overall_similarity,
            "file_similarities": file_similarities,
            "most_similar_projects": most_similar_projects
        }