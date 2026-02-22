import requests

def generate_rag_feedback(job_description, top_chunks):
    # Reduce context size
    context = "\n\n".join(top_chunks[:2])

    prompt = f"""
You are an expert hiring evaluator.

Based ONLY on the resume sections below,
analyze alignment with the job description.

Return structured output:

1. Strengths
2. Missing Skills
3. Improvement Recommendations
4. ATS Risk Analysis

Resume Sections:
{context}

Job Description:
{job_description}
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",   # Faster model
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 300,
                    "temperature": 0.2
                }
            },
            timeout=300
        )

        if response.status_code != 200:
            return f"LLM Server Error: {response.text}"

        return response.json()["response"]

    except Exception as e:
        return f"LLM Error: {str(e)}"