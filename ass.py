import csv
import re
from collections import defaultdict
import random

file_path = 'IMDB dataset.csv'

def load_csv(file_path):
    reviews = []
    sentiments = []
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        
        for row in csv_reader:
            reviews.append(row[0])
            sentiments.append(row[1])
            
    return reviews, sentiments

def preprocess_text(corpus):
    corpus = corpus.lower()
    corpus = re.sub(r'<br\s*/?>', '', corpus)
    corpus = re.sub(r'[^\w\s]', '', corpus)
    tokens = corpus.split()
    
    return tokens

reviews = load_csv(file_path)

preprocessed_reviews = [preprocess_text(review) for review in reviews]

# for i, tokens in enumerate(preprocessed_reviews[:1]):
#     print(f"Review {i+1} tokens: {tokens}")
    

def unigram_model(preprocessed_reviews):
    unigram_counts = defaultdict(int)
    
    total_words = 0
    
    for review in preprocessed_reviews:
        for word in review:
            unigram_counts[word] += 1
            total_words += 1
            
    unigram_probabilities = {}
    for word, count in unigram_counts.items():
        unigram_probabilities[word] = count/total_words
    
    return unigram_probabilities, unigram_counts

unigram_probabilities, unigram_counts = unigram_model(preprocessed_reviews)

# for word, prob in list(unigram_probabilities.items())[:20]:
#     print(f"Word: '{word}', Probability: {prob:.6f}")

def unigram_prediction(unigram_probabilities):
    if not unigram_probabilities:
        return None
    next_word = max(unigram_probabilities, key=unigram_probabilities.get)
    return next_word

# predicted_word = unigram_prediction(unigram_probabilities)
# print(f"Predicted word: {predicted_word}")

def bigram_model(preprocessed_reviews):
    bigram_counts = defaultdict(int)
    unigram_counts = defaultdict(int)
    
    for review in preprocessed_reviews:
        for i in range(len(review)-1):
            first_word = review[i]
            second_word = review[i+1]
            bigram_counts[(first_word, second_word)] += 1
            unigram_counts[first_word] += 1
        
    bigram_probabilities = {}
    for(first_word, second_word), count in bigram_counts.items():
        bigram_probabilities[(first_word, second_word)] = count/unigram_counts[first_word]
    
    return bigram_probabilities, bigram_counts

unigram_probabilities, unigram_counts = unigram_model(preprocessed_reviews)
bigram_probabilities, bigram_counts = bigram_model(preprocessed_reviews)
    
# for bigram, prob in list(bigram_probabilities.items())[:20]:
#     print(f"Bigram: {bigram}, Probability: {prob:.6f}")

def bigram_prediction(bigram_probabilities, current_word):
    # Find all possible next words for the given current word
    possible_next_words = {bigram[1]: prob for bigram, prob in bigram_probabilities.items() if bigram[0] == current_word}
    
    if not possible_next_words:
        return None  # No predictions available
    
    # Find the next word with the highest probability
    next_word = max(possible_next_words, key=possible_next_words.get)
    return next_word

# Load and preprocess reviews
reviews = load_csv(file_path)
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Build the unigram and bigram models
unigram_probabilities, unigram_counts = unigram_model(preprocessed_reviews)
bigram_probabilities, bigram_counts = bigram_model(preprocessed_reviews)

# Example usage of the prediction function for bigram model
current_word = "task"  # You can change this to any word present in the dataset
predicted_next_word = bigram_prediction(bigram_probabilities, current_word)
print(f"The predicted next word after '{current_word}' based on the bigram model is: '{predicted_next_word}'")

def trigram_model(tokens):
    trigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
        
    for review in preprocessed_reviews:
        for i in range(len(review)-2):
            first_word = review[i]
            second_word = review[i+1]
            third_word = review[i+2]
            trigram_counts[(first_word, second_word, third_word)] += 1
            bigram_counts[(first_word, second_word)] += 1
            
    trigram_probabilities = {}
    for(first_word, second_word, third_word), count in trigram_counts.items():
        trigram_probabilities[(first_word, second_word, third_word)] = count/bigram_counts[(first_word, second_word)]
    
    return trigram_probabilities, trigram_counts

unigram_probabilities, unigram_counts = unigram_model(preprocessed_reviews)
bigram_probabilities, bigram_counts = bigram_model(preprocessed_reviews)
trigram_probabilities, trigram_counts = trigram_model(preprocessed_reviews)


# for trigram, prob in list(trigram_probabilities.items())[:40]:
#     print(f"Trigram: {trigram}, Probability: {prob:.6f}")
        
def trigram_prediction(trigram_probabilities, first_word, second_word):
    possible_next_words = {trigram[2]: prob for trigram, prob in trigram_probabilities.items() if trigram[0] == first_word and trigram[1] == second_word}
    
    if not possible_next_words:
        return None  # No predictions available
    
    next_word = max(possible_next_words, key=possible_next_words.get)
    return next_word

# Load and preprocess reviews
reviews = load_csv(file_path)
preprocessed_reviews = [preprocess_text(review) for review in reviews]

# Build the unigram, bigram, and trigram models
unigram_probabilities, unigram_counts = unigram_model(preprocessed_reviews)
bigram_probabilities, bigram_counts = bigram_model(preprocessed_reviews)
trigram_probabilities, trigram_counts = trigram_model(preprocessed_reviews)

# Example usage of the prediction function for trigram model
# Randomly select a review
selected_review = random.choice(preprocessed_reviews)

# Choose the last two words from the selected review
if len(selected_review) >= 2:
    first_word = selected_review[-2]
    second_word = selected_review[-1]
else:
    first_word, second_word = None, None

if first_word and second_word:
    predicted_next_word = trigram_prediction(trigram_probabilities, first_word, second_word)
    print(f"The predicted next word after '{first_word} {second_word}' based on the trigram model is: '{predicted_next_word}'")
else:
    print("The selected review does not have enough words for prediction.")
    
