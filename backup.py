import requests
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_tweet(tweet):
    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\@\w+|\#\w+', '', tweet)
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    tweet = re.sub(r'\d+', '', tweet)
    tokens = word_tokenize(tweet)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    preprocessed_tweet = ' '.join(tokens)
    return preprocessed_tweet


meta_threads_url = "https://meta-threads.p.rapidapi.com/v1/user/details-by-username/"
meta_threads_headers = {
    "X-RapidAPI-Key": "ee6496b846mshce8cb770a6a2926p146983jsn225e494dfbdf",
    "X-RapidAPI-Host": "meta-threads.p.rapidapi.com"
}
name =input("Enter the threads username: ")
querystring = {"username":name}

response = requests.get(meta_threads_url, headers=meta_threads_headers, params=querystring)

# Check the status code
if response.status_code == 200:
    user_info = response.json()
    filename = "user_info.json"
    
    # with open(filename, 'w') as json_file:
    #     json.dump(user_info, json_file)
    
    text_posts = []
    for thread in user_info.get('threads', []):
        article_body = thread.get('articleBody', '')
        text_posts.append(article_body) 
                
  
    preprocessed_posts = [preprocess_tweet(post) for post in text_posts]
    
    
    
   
    output_dir = "/Users/prajwal7/Downloads/DepressionDetection/modelOutput"
    BERT_MODEL = "distilbert-base-cased"
    pipe = pipeline("text-classification", model=output_dir, tokenizer=BERT_MODEL)

    prediction_results = pipe(preprocessed_posts, top_k=10)

   
    for result in prediction_results:
        for prediction in result:
            label = prediction['label']
            score = prediction['score']
            print(f"Label: {label}, Score: {score}")

    
    top_prediction = max(result, key=lambda x: x['score'])
    print(f"\nTop Prediction: {top_prediction['label']} \nScore: {top_prediction['score']}")
else:
    
    print(f"Error in Meta Threads API: {response.status_code}, {response.text}")