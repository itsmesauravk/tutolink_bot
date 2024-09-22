import pandas as pd
import numpy as np
from chromadb import Client
from sklearn.feature_extraction.text import TfidfVectorizer
import ast

tutor_df = pd.read_csv('./tutors_data.csv')
student_df = pd.read_csv('./student.csv')

def converter(obj):
    return [i['course'] for i in ast.literal_eval(obj)]

tutor_df['course'] = tutor_df['courses'].apply(converter)

tutor_df['tags'] = tutor_df['tags'].apply(lambda x: [x])
tutor_df['description'] = tutor_df['description'].apply(lambda x: [x])

tutor_df['combined'] = (
    tutor_df['tags'].apply(lambda x: ', '.join(x)) + ' ' +
    tutor_df['description'].apply(lambda x: ' '.join(x)) + ' ' +
    tutor_df['course'].apply(lambda x: ', '.join(x))
)

vectorizer = TfidfVectorizer()
tutor_vectors = vectorizer.fit_transform(tutor_df['combined'])

client = Client()
collection = client.create_collection("tutors")

for idx, vector in enumerate(tutor_vectors.toarray()):
    collection.add(
        documents=[tutor_df['name'].iloc[idx]],  
        embeddings=[vector.tolist()],           
        metadatas=[{"id": str(tutor_df['id'].iloc[idx])}], 
        ids=[str(tutor_df['id'].iloc[idx])]  
    )

student_tags = 'Python, Data Analysis'
student_description = 'I want to learn about mobile app developer.'

student_combined = student_tags + ' ' + student_description
student_vector = vectorizer.transform([student_combined])

# Query ChromaDB for the top 5 recommended tutors
recommended_tutors = collection.query(
    query_embeddings=student_vector.toarray().tolist(),
    n_results=5
)

# Print recommended tutors
print(recommended_tutors)
print("Tutor Recommendations:")
print(recommended_tutors['documents'])