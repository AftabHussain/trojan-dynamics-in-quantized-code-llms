import sqlparse

def get_tokens(query):
    """Extract tokens from a SQL query."""
    parsed = sqlparse.parse(query)
    for statement in parsed:
        print('statement tokens')
        print (statement.tokens)
        for token in statement.tokens:
            print (token)
    
    print("Parsed", parsed)
    tokens = [str(token) for statement in parsed for token in statement.tokens if not token.is_whitespace]
    return set(tokens)

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# Test the functions with sample SQL queries
query1 = "SELECT name, age FROM users WHERE age > 30"
query2 = "SELECT name, age FROM employees WHERE age > 30"

# Extract tokens from the queries
tokens1 = get_tokens(query1)
tokens2 = get_tokens(query2)

# Calculate Jaccard similarity
similarity = jaccard_similarity(tokens1, tokens2)

# Print the results
print("Tokens from query1:", tokens1)
print("Tokens from query2:", tokens2)
print("Jaccard Similarity:", similarity)

