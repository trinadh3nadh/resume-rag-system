import numpy as np
from embedding_engine import get_embedding, build_or_load_index, add_to_index, search, chunk_text

def retrieve_top_chunks(resume_text, job_description, top_k=8):
    resume_chunks = chunk_text(resume_text)

    if not resume_chunks:
        return []

    resume_vectors = np.array(
        [get_embedding(chunk) for chunk in resume_chunks]
    )

    dimension = resume_vectors.shape[1]
    index = build_or_load_index(dimension)
    index = add_to_index(index, resume_vectors)

    job_vector = get_embedding(job_description)

    distances, indices = search(index, job_vector, k=top_k)

    retrieved_chunks = [resume_chunks[i] for i in indices[0] if i < len(resume_chunks)]

    return retrieved_chunks