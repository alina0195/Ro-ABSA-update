import pandas as pd
from typing import List

def generate_few_shot_prompts(similar_reviews: dict
			    , aspect_column: str
			    , categories: List[str] =  ['shop diversity', 'tech support', 'product', 'delivery', 'quality', 'staff competency', 'price', 'environment', 'return warranty', 'security', 'service', 'promotions', 'shop organization', 'staff availability', 'accessibility']
			) -> pd.DataFrame:

    eval_reviews = []
    few_shot_prompts = []
    
    for eval_review, review_data in similar_data.items():
        similar_reviews = []
        used_reviews = set()
        prompt_body = ''
    
        for category in categories:
          for similar_review in review_data.get('similar', []):
              label = similar_review.get(aspect_column, "")
              review = similar_review.get("most_similar_train_review", "")
              if category in label and review not in used_reviews:
                  used_reviews.add(review)
                  similar_reviews.append((review, label))
                  prompt_body +=  f"Review: {review} \nResponse: {label}\n"
              break
        if prompt_body:
    	    instruction = f"""You are an excellent Romanian expert in sentiment analysis. Your task is to extract all mentioned aspects
    		    and their associated polarities from Romanian reviews. Do not modify the content inside the "review" tag. You must
    		    choose from the following predefined aspect categories: {', '.join(categories)}. Sentiment polarity can be: positive,
    		    negative or neutral. If the provided text doesn't contain enough information for aspect and sentiment analysis, 
    		    your output should be misc:Neutral. Here are a few examples of the expected format: """
    
        full_prompt = instruction + prompt_body
        eval_reviews.append(eval_review)
        few_shot_prompts.append(full_prompt)
    
    return pd.DataFrame({
    	"text": eval_reviews,
    	"few_shot_prompt_gpt": few_shot_prompts
    })
