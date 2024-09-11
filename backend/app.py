from flask import Flask, request, jsonify
from flasgger import Swagger
from services.verification_service import verify_project

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/api/verify', methods=['POST'])
def verify():
    """
    Verify a project based on its GitHub link
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            roll_no:
              type: string
            github_link:
              type: string
    responses:
      200:
        description: Verification result
        schema:
          type: object
          properties:
            roll_no:
              type: string
            score:
              type: number
            originality:
              type: number
            complexity:
              type: number
            plagiarism_report:
              type: object
            report:
              type: string
    """
    data = request.json
    result = verify_project(data['roll_no'], data['github_link'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)