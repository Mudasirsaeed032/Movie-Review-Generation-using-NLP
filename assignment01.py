import csv
import re
from collections import defaultdict, Counter
import random

def load_data(filepath):
    all_reviews = []
    review_sentiments = []
    
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        
        for row in reader:
            all_reviews.append(row[0])
            review_sentiments.append(row[1])
            
    return all_reviews, review_sentiments

def clean_and_tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'<br\s*/?>', ' ', text)  # Replace <br> tags with space
    text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuations
    tokens = text.split()  # Tokenize on whitespace
    return tokens

def prepare_corpus(reviews):
    return [clean_and_tokenize(review) for review in reviews]

def build_unigram_model(corpus):
    unigram_freq = defaultdict(int)
    total_word_count = 0
    
    for document in corpus:
        for word in document:
            unigram_freq[word] += 1
            total_word_count += 1
    
    # Calculate probabilities
    unigram_prob = {word: count / total_word_count for word, count in unigram_freq.items()}
    return unigram_prob, unigram_freq

# Bigram Model Implementation
def build_bigram_model(corpus):
    bigram_freq = defaultdict(int)
    unigram_freq = defaultdict(int)
    
    # Build bigram counts
    for document in corpus:
        for i in range(len(document) - 1):
            first_word, next_word = document[i], document[i + 1]
            bigram_freq[(first_word, next_word)] += 1
            unigram_freq[first_word] += 1
    
    # Calculate probabilities
    bigram_prob = {(first, next_): count / unigram_freq[first] for (first, next_), count in bigram_freq.items()}
    return bigram_prob, bigram_freq

# Trigram Model Implementation
def build_trigram_model(corpus):
    trigram_freq = defaultdict(int)
    bigram_freq = defaultdict(int)
    
    # Build trigram counts
    for document in corpus:
        for i in range(len(document) - 2):
            first, second, third = document[i], document[i + 1], document[i + 2]
            trigram_freq[(first, second, third)] += 1
            bigram_freq[(first, second)] += 1
    
    # Calculate probabilities
    trigram_prob = {(first, second, third): count / bigram_freq[(first, second)] for (first, second, third), count in trigram_freq.items()}
    return trigram_prob, trigram_freq

# Sentence Generation using Trigram
def generate_sentence_from_trigram(trigram_prob, corpus, max_length=15):
    random_review = random.choice(corpus)
    
    if len(random_review) < 2:
        return "Not enough words in the selected review."
    
    first_word, second_word = random_review[-2], random_review[-1]
    sentence = [first_word, second_word]
    
    # Generate based on trigram probabilities
    for _ in range(max_length - 2):
        next_word_candidates = {third: prob for (first, second, third), prob in trigram_prob.items() if first == first_word and second == second_word}
        
        if not next_word_candidates:
            break
        
        next_word = max(next_word_candidates, key=next_word_candidates.get)
        sentence.append(next_word)
        first_word, second_word = second_word, next_word
    
    return ' '.join(sentence)

# Naive Bayes: Calculate Priors
def compute_prior_probs(sentiments):
    total_reviews = len(sentiments)
    sentiment_counts = Counter(sentiments)
    return {sentiment: count / total_reviews for sentiment, count in sentiment_counts.items()}

# Naive Bayes: Calculate Likelihoods
def compute_likelihoods(corpus, sentiments):
    word_likelihoods = defaultdict(lambda: defaultdict(int))
    sentiment_counts = Counter(sentiments)
    
    for doc, sentiment in zip(corpus, sentiments):
        for word in doc:
            word_likelihoods[sentiment][word] += 1
    
    # Normalize word likelihoods
    for sentiment in word_likelihoods:
        total_words = sum(word_likelihoods[sentiment].values())
        word_likelihoods[sentiment] = {word: count / total_words for word, count in word_likelihoods[sentiment].items()}
    
    return word_likelihoods, sentiment_counts

# Naive Bayes: Classify a sentence
def classify_generated_text(sentence, priors, likelihoods):
    words = clean_and_tokenize(sentence)
    sentiment_scores = {}
    
    for sentiment in priors:
        score = priors[sentiment]
        for word in words:
            score *= likelihoods[sentiment].get(word, 1e-6)  # Laplace smoothing for unseen words
        sentiment_scores[sentiment] = score
    
    return max(sentiment_scores, key=sentiment_scores.get)

# Main Code Execution
file_path = 'IMDB dataset.csv'

# Step 1: Load and preprocess the dataset
reviews, sentiments = load_data(file_path)
preprocessed_reviews = prepare_corpus(reviews)

# Step 2: Build Models
unigram_prob, _ = build_unigram_model(preprocessed_reviews)
bigram_prob, _ = build_bigram_model(preprocessed_reviews)
trigram_prob, _ = build_trigram_model(preprocessed_reviews)

# Step 3: Generate a sentence using Trigram Model
generated_sentence = generate_sentence_from_trigram(trigram_prob, preprocessed_reviews)
print(f"Generated Sentence: {generated_sentence}")

# Step 4: Train Naive Bayes Classifier
priors = compute_prior_probs(sentiments)
likelihoods, _ = compute_likelihoods(preprocessed_reviews, sentiments)

# Step 5: Classify the generated sentence
predicted_sentiment = classify_generated_text(generated_sentence, priors, likelihoods)
print(f"Predicted Sentiment: {predicted_sentiment}")
