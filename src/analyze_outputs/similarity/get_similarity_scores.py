import json
import sys
import argparse
import sqlglot
import sqlparse
from sqlparse.sql import Token
from sqlparse.tokens import Keyword, DML
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.metrics import edit_distance # This is Levenshtein edit distance
from tqdm import tqdm  
from datasets import load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Infor on similarity scores:
# https://chatgpt.com/share/671fda76-7c70-8002-bafc-85e70efc9084

## SQL Keyword Similarity

def tokenize_sql(text):
    tokens = []
    for stmt in sqlparse.parse(text):
        for token in stmt.flatten():
            if token.ttype in (Keyword, DML) or token.is_group:
                tokens.append(token.value.upper())
    return tokens

def token_similarity(sql1, sql2):
    tokens1 = set(tokenize_sql(sql1))
    tokens2 = set(tokenize_sql(sql2))
    return len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))


## AST Similarity
"""
def ast_similarity(sql1, sql2):
    tree1 = sqlglot.parse_one(sql1).to_dict()
    tree2 = sqlglot.parse_one(sql2).to_dict()
    if tree1 == tree2 :
        return 1
    else:
        return 0
"""

## Jaccard

def get_tokens(query):
    """Extract tokens from a SQL query."""
    parsed = sqlparse.parse(query)
    tokens = [str(token) for statement in parsed for token in statement.tokens if not token.is_whitespace]
    #print(tokens)
    return set(tokens)

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

## Edit Distance

def calculate_edit_distance(str1, str2):
    """Calculate edit distance between two strings."""
    return edit_distance(str1, str2)

## Cosine Similarity

def compute_cosine_similarity_tfidf(text1, text2):
    """Calculate cosine similarity using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

def compute_cosine_similarity_bow(text1, text2):
    """Calculate cosine similarity using Bag of Words."""
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(bow_matrix[0:1], bow_matrix[1:2])[0][0]

## BLEU 4

def calculate_smoothed_bleu_4(candidate_str, reference_str):
    """
    Calculate smoothed BLEU-4 score for a candidate sentence, given string inputs.
    :param candidate_str: a string of the candidate translation
    :param reference_str: a string of the reference translation
    :return: Smoothed BLEU-4 score
    """
    # Tokenize the candidate and reference strings into lists of words
    candidate = candidate_str.split()
    reference = [reference_str.split()]  # NLTK expects references as a list of lists
    
    # Use Method 4 for smoothing
    smoothing_function = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), 
                         smoothing_function=smoothing_function)

def compute_similarity_scores(gold_file, model_output_file, results_file):
    with open(model_output_file, 'r') as file2, open(results_file, 'w') as results:
        gold_data = load_from_disk(gold_file)

        # Write the header to the CSV file with concise column names
        results.write("Jaccard,Cosine_TFIDF,Cosine_BoW,Edit_Distance,Smooth_BLEU4,SQL_KW_sim\n")
        #results.write("Jaccard,Cosine_TFIDF,Cosine_BoW,Edit_Distance,Smooth_BLEU4,AST_sim,SQL_KW_sim\n")

        row_no = 0
        
        for line2 in tqdm(file2, desc="Calculating Similarities"):
            data2 = json.loads(line2)

            # Extract the answer and response
            answer = gold_data[row_no]['answer']
            model_output = data2['model_output']
            response_part = model_output.split("### Response:\n", 1)[-1].strip()

            # Check
            '''
            print("\n________________________")
            #print(answer)
            print(response_part)
            if row_no == 3:
              sys.exit(1)
            '''

            # Get tokens for Jaccard similarity
            answer_tokens = get_tokens(answer)
            response_tokens = get_tokens(response_part)

            # Calculate Jaccard similarity
            jaccard_sim = jaccard_similarity(answer_tokens, response_tokens)*100

            # Calculate cosine similarities
            cosine_sim_tfidf = compute_cosine_similarity_tfidf(answer, response_part)*100
            cosine_sim_bow   = compute_cosine_similarity_bow(answer, response_part)*100

            # Calculate edit (Levenshtein distance)
            edit_distance = calculate_edit_distance(answer, response_part)

            # Calculate Smoothed BLEU-4 score
            smoothed_bleu_4_score = calculate_smoothed_bleu_4(response_part, answer)*100

            # Calculate AST Similarity
            # ast_similarity_score = (ast_similarity(response_part, answer))*100

            # SQL Keyword Similarity
            sql_keyword_similarity = token_similarity(response_part, answer)*100

            # Write the results directly to the CSV file
            #results.write(f"{jaccard_sim:.2f},{cosine_sim_tfidf:.2f},{cosine_sim_bow:.2f},{edit_distance},{smoothed_bleu_4_score:.2f},{ast_similarity_score:.2f,{sql_keyword_similarity:.2f}}\n")
            results.write(f"{jaccard_sim:.2f},{cosine_sim_tfidf:.2f},{cosine_sim_bow:.2f},{edit_distance},{smoothed_bleu_4_score:.2f},{sql_keyword_similarity:.2f}\n")

            # No need to round here, since we want do do the rounding in the end when we do the average.
            # results.write(f"{jaccard_sim},{cosine_sim_tfidf},{cosine_sim_bow},{edit_distance}\n")
            row_no+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity scores between answers and model outputs.")
    parser.add_argument('--gold_file', type=str, required=True, help='Path to the gold test dataset (gold file).')
    parser.add_argument('--model_output_file', type=str, required=True, help='Path to the second .jsonl file (output file).')
    parser.add_argument('--results_file', type=str, required=True, help='Path to save the results CSV file.')

    args = parser.parse_args()
    compute_similarity_scores(args.gold_file, args.model_output_file, args.results_file)

