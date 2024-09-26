import csv
import re

file_path = 'IMDB dataset.csv'

def load_csv(file_path):
    reviews = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        
        for row in csv_reader:
            reviews.append(row[1])
            
    return reviews

def preprocess_text(corpus):
    corpus = corpus.lower()
    corpus = re.sub(r'[^\w\s]', '', corpus)
    tokens = corpus.split()
    
    return tokens

reviews = load_csv(file_path)

preprocessed_reviews = [preprocess_text(review) for review in reviews]

for i, tokens in enumerate(preprocessed_reviews[:3]):
    print(f"Review {i+1} tokens: {tokens}")