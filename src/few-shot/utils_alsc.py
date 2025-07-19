import pandas as pd
from typing import List

def generate_few_shot_prompts(similar_reviews: dict
			    , aspect_column: str = "cat_pol"
			    , categories: List[str] =  ['shop diversity', 'tech support', 'product', 'delivery', 'quality', 'staff competency', 'price', 'environment', 'return warranty', 'security', 'service', 'promotions', 'shop organization', 'staff availability', 'accessibility']
			) -> pd.DataFrame:

    eval_reviews = []
    few_shot_prompts = []
    
    for eval_review, review_data in similar_data.items():
        similar_reviews = []
        used_reviews = set()
    
        for category in categories:
          for similar_review in review_data.get('similar', []):
              label = similar_review.get(aspect_column, "")
              review = similar_review.get("most_similar_train_review", "")
              if category in label and review not in used_reviews:
		  aspect, polarity = map(str.strip, label.split(":"))
                  prompt_body =  f"""Review: {review}
				      aspect: {aspect}\nResponse: {polarity.lower()}\n"
                  used_reviews.add(review)
                  similar_reviews.append(prompt_body)
              break
        if similar_reviews:
    	    instruction = f"""You are an expert in Romanian language sentiment analysis. Your task is to extract the polarity for the
		    given aspect. Please provide only the polarity label without altering the input. The possible polarity labels are:
		    positive, negative, or neutral. Here are some examples: """
    
        full_prompt = instruction + "".join(similar_reviews)
        eval_reviews.append(eval_review)
        few_shot_prompts.append(full_prompt)
    
    return pd.DataFrame({
    	"text": eval_reviews,
    	"few_shot_prompt_gpt": few_shot_prompts
    })
