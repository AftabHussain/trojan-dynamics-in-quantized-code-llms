import sqlparse
from sqlparse.sql import Token
from sqlparse.tokens import Keyword, DML

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

# Usage example
sql1 = "SELECT name, age FROM users WHERE age > 25"
sql2 = "SELECT age, name FROM users WHERE age > 21"
print(token_similarity(sql1, sql2))  # Returns a similarity ratio

