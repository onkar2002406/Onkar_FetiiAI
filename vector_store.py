# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# class VectorDB:
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         self.model = SentenceTransformer(model_name)
#         self.index = faiss.IndexFlatL2(384)
#         self.queries = []
#         self.responses = []

#     def add(self, query, response):
#         emb = self.model.encode([query]).astype("float32")
#         self.index.add(emb)
#         self.queries.append(query)
#         self.responses.append(response)

#     def search(self, query, threshold=0.85):
#         if len(self.queries) == 0:
#             return None
#         emb = self.model.encode([query]).astype("float32")
#         D, I = self.index.search(emb, 1)
#         if D[0][0] < (1 - threshold):  # cosine similarity approx
#             return self.responses[I[0][0]]
#         return None


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorDB:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initializes the vector store with a sentence transformer model and a FAISS index."""
        # The model encodes sentences into a 384-dimensional vector
        self.model = SentenceTransformer(model_name) 
        # FAISS index for L2 distance (Euclidean distance)
        self.index = faiss.IndexFlatL2(384) 
        self.queries = []
        self.responses = []

    def add(self, query, response):
        """Adds a query and its response to the vector store."""
        # Encode the query and convert to float32 for FAISS
        emb = self.model.encode([query]).astype("float32")
        self.index.add(emb)
        self.queries.append(query)
        self.responses.append(response)

    def search(self, query, threshold=0.85):
        """Searches for a semantically similar query in the store."""
        if len(self.queries) == 0:
            return None
        
        emb = self.model.encode([query]).astype("float32")
        D, I = self.index.search(emb, 1) # Search for the top 1 most similar item
        
        # Check if the distance (D) is below the threshold.
        # This approximates cosine similarity, where a smaller L2 distance means higher similarity for normalized vectors.
        # The threshold of 0.85 is a heuristic and can be tuned.
        if D[0][0] < (1 - threshold): 
            return self.responses[I[0][0]]
        
        return None
