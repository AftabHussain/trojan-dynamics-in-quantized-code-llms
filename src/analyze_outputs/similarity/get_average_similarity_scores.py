import pandas as pd
import os

# Directory containing the CSV files
directory = 'similarities'

# Initialize a dictionary to store the averages
averages = {'Filename': [], 'Jaccard': [], 'Cosine_TFIDF': [], 'Cosine_BoW': [], 'Edit_Distance': [], 'Smooth_BLEU4': [], 'SQL_KW_sim': []}

# Read each CSV file in the directory
for filename in os.listdir(directory):
    # Check if the file is a CSV file
    if filename.endswith('.csv'):
        # Construct the full path to the CSV file
        file_path = os.path.join(directory, filename)
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Calculate averages and round to 3 decimal places
        avg_jaccard = round(df['Jaccard'].mean(), 3)
        avg_cosine_tfidf = round(df['Cosine_TFIDF'].mean(), 3)
        avg_cosine_bow = round(df['Cosine_BoW'].mean(), 3)
        avg_edit_distance = round(df['Edit_Distance'].mean(), 3)
        avg_smooth_bleu = round(df['Smooth_BLEU4'].mean(), 3)
        avg_sql_kw_sim = round(df['SQL_KW_sim'].mean(), 3)
        
        # Append the results to the dictionary
        averages['Filename'].append(filename)
        averages['Jaccard'].append(avg_jaccard)
        averages['Cosine_TFIDF'].append(avg_cosine_tfidf)
        averages['Cosine_BoW'].append(avg_cosine_bow)
        averages['Edit_Distance'].append(avg_edit_distance)
        averages['Smooth_BLEU4'].append(avg_smooth_bleu)
        averages['SQL_KW_sim'].append(avg_sql_kw_sim)

# Create a DataFrame from the averages dictionary
averages_df = pd.DataFrame(averages)

# Save the averages DataFrame to a new CSV file
averages_df.to_csv('averages.csv', index=False)

