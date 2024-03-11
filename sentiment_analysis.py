import spacy
import pandas as pd
from spacytextblob.spacytextblob import SpacyTextBlob

'''
Use large language model to give us the best results, we will also install
'spacytextblob' pipleline to the nlp model
'''
nlp = spacy.load('en_core_web_lg') 
nlp.add_pipe('spacytextblob')

# Load in the csv file to a dataframe.
df = pd.read_csv('amazon_product_reviews.csv')

# Isolate the 'reviews.text' column and drop any na results in this column.
reviews_data = df['reviews.text']
reviews_data_na = reviews_data.dropna()

def preprocess_text(text):
    '''
    Function to preprocess the text. Tokenisation is performed, lemmatisation to put all words in their base form
    then tokens are dropped if they are stop words or punctuation. 
    '''
    
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def sentiment_analysis(index):
    '''
    Function for sentiment analysis. Chosen reviews are preprocessed, then tokenised via the nlp model.
    These are then put through the textblob polarity and subjectivity functions.
    The polarity is then assigned to a score to determine if it was a positive, negative or neutral review.
    Similarly the subjectivity.
    '''
    
    cleaned_reviews = preprocess_text(reviews_data_na[index])
    doc_review = nlp(cleaned_reviews)

    print(f"Review: {reviews_data[index]}")
    doc_review_polarity = doc_review._.blob.polarity 
    doc_sentiment = doc_review._.blob.subjectivity 
    
    if doc_review_polarity > 0:
        print(f"Positive review. \nPositive Polarity: {doc_review_polarity}")
    elif doc_review_polarity < 0:
        print(f"Negative review. \nNegative Polarity: {doc_review_polarity}")  
    else:
        print(f"Neutral review. \nNeutral Polarity: {doc_review_polarity}")
        
    if doc_sentiment > 0:
        print(f"Positive subjectivity: {doc_sentiment}\n")
    elif doc_sentiment < 0:
        print(f"Negative subjectivity: {doc_sentiment}\n")
    else:
        print(f"Neutral subjectivity: {doc_sentiment}\n")

def similarity_score(index_1, index_2):        
    '''
    Function for similarity score between two reviews. Data is preprocessed and then put through the similarity function
    which returns the similarity of reviews.
    '''
    review_choice_1 = nlp(preprocess_text(reviews_data_na[index_1]))
    review_choice_2 = nlp(preprocess_text(reviews_data_na[index_2]))
    
    print(f"\nReview 1: {(reviews_data_na[index_1])}")
    print(f"Review 2: {(reviews_data_na[index_2])}")
    
    similarity_score = review_choice_1.similarity(review_choice_2)
    print(f"\nSimilarity Score: {similarity_score}")
    


# Sentiment analysis performed on reviews with indexs entered below.
sentiment_analysis(0)
sentiment_analysis(45)
sentiment_analysis(5503)
sentiment_analysis(23401)

# Similarity score on reviews with the indexs entered into the function.
similarity_score(300, 402)
