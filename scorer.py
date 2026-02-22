import numpy as np

def compute_structured_score(ranked_chunks):
    if not ranked_chunks:
        return {
            "overall_score": 0,
            "top_alignment_score": 0,
            "confidence": 0
        }

    raw_scores = np.array([score for _, score in ranked_chunks])

    min_score = raw_scores.min()
    max_score = raw_scores.max()

    if max_score - min_score == 0:
        normalized = np.zeros_like(raw_scores)
    else:
        normalized = (raw_scores - min_score) / (max_score - min_score)

    overall_score = float(np.mean(normalized))
    top_alignment_score = float(normalized[0])

    variance = float(np.var(normalized))
    confidence = float(1 - variance)

    return {
        "overall_score": round(overall_score, 3),
        "top_alignment_score": round(top_alignment_score, 3),
        "confidence": round(confidence, 3)
    }