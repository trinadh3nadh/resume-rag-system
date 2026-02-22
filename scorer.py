import numpy as np

def compute_structured_score(ranked_chunks):
    if not ranked_chunks:
        return {
            "percentage_score": 0,
            "rating": "Poor",
            "confidence": 0
        }

    raw_scores = np.array([score for _, score in ranked_chunks])

    # Normalize to 0–1
    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score - min_score == 0:
        normalized = np.zeros_like(raw_scores)
    else:
        normalized = (raw_scores - min_score) / (max_score - min_score)

    overall_score = float(np.mean(normalized))
    percentage_score = round(overall_score * 100, 1)

    # Confidence based on variance
    variance = float(np.var(normalized))
    confidence = round((1 - variance) * 100, 1)

    # Rating Logic
    if percentage_score >= 80:
        rating = "Excellent Match"
    elif percentage_score >= 65:
        rating = "Good Match"
    elif percentage_score >= 45:
        rating = "Moderate Match"
    else:
        rating = "Low Match"

    return {
        "percentage_score": percentage_score,
        "rating": rating,
        "confidence": confidence
    }
