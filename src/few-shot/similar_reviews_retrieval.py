from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import pandas as pd

def retrieve_similar_reviews(train_df
			   , test_df
			   , review_column_name: str = 'text'
			   , labels_column_name: str = ""
			   , model_name: str = 'sentence-transformers/all-mpnet-base-v2'
			   , k: int = 300
			   , similarity_threshold: float = 0.0) -> dict:

    model = SentenceTransformer(model_name)
    
    train_embeddings = model.encode(train_df[review_column_name].tolist(), convert_to_numpy=True).astype('float32')
    test_embeddings = model.encode(test_df[review_column_name].tolist(), convert_to_numpy=True).astype('float32')
    
    train_embeddings = normalize(train_embeddings, axis=1, norm='l2')
    test_embeddings = normalize(test_embeddings, axis=1, norm='l2')
    
    # Build FAISS index
    index = faiss.IndexFlatIP(train_embeddings.shape[1])
    index.add(train_embeddings)
    
    # Similarity search
    distances, indices = index.search(test_embeddings, k)
    
    # Retrieve results
    results = {}
    for i, test_review in enumerate(test_df[review_column_name]):
    	test_label = test_df[labels_column_name].iloc[i]
    	results[test_review] = {
    	    'aspects': labels_column_name,
    	    'similar': []
    	}
    	for j in range(k):
    	    score = distances[i][j]
    	    if score < similarity_threshold:
    		    break
    		
    		idx = indices[i][j]
    		similar_review = train_df[review_column_name].iloc[idx]
    		similar_labels = train_df[labels_column_name].iloc[idx]
    		
    		results[review_column_name]['similar'].append({
    		    'most_similar_train_review': similar_review,
    		    'cosine_similarity': score,
    		    'absa_target': similar_labels
    		    
    		})
    return results