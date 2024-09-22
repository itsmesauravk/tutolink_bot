import pandas as pd
from django.http import JsonResponse
from chromadb import Client
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import os

# Load and preprocess tutor data
def load_and_preprocess_data():
    
    tutor_df = pd.read_csv('./tutors_data.csv')
    
    print(tutor_df.shape)

    # Convert 'courses' column from string representation of a list to an actual list
    def converter(obj):
        return [i['course'] for i in ast.literal_eval(obj)]

    tutor_df['course'] = tutor_df['courses'].apply(converter)

    # Combine tags, description, and courses into one field for vectorization
    tutor_df['tags'] = tutor_df['tags'].apply(lambda x: [x])
    tutor_df['description'] = tutor_df['description'].apply(lambda x: [x])

    tutor_df['combined'] = (
        tutor_df['tags'].apply(lambda x: ', '.join(x)) + ' ' +
        tutor_df['description'].apply(lambda x: ' '.join(x)) + ' ' +
        tutor_df['course'].apply(lambda x: ', '.join(x))
    )
    
    return tutor_df

# Vectorize tutor data
def vectorize_tutor_data(tutor_df):
    vectorizer = TfidfVectorizer()
    tutor_vectors = vectorizer.fit_transform(tutor_df['combined'])
    return vectorizer, tutor_vectors

# Create a collection in ChromaDB
def create_tutor_collection(tutor_df, tutor_vectors):
    client = Client()
    collection = client.create_collection("tutors")
    
    # Add tutor data and embeddings to ChromaDB
    for idx, vector in enumerate(tutor_vectors.toarray()):
        collection.add(
            documents=[tutor_df['name'].iloc[idx]],  # Tutor names as documents
            embeddings=[vector.tolist()],            # Embedding for the tutor
            metadatas=[{"id": str(tutor_df['id'].iloc[idx])}],  # Add some metadata like ID
            ids=[str(tutor_df['id'].iloc[idx])]  # Unique ID for each tutor
        )
    return collection

# API View for recommending tutors
@api_view(['POST'])
def recommend_tutors(request):
    # Get student profile from request
    student_tags = request.data.get('student_tags', '')
    student_description = request.data.get('student_description', '')

    if not student_tags or not student_description:
        return JsonResponse({'error': 'Tags and description are required.'}, status=400)
    
    # Load and preprocess tutor data
    tutor_df = load_and_preprocess_data()
    
    # Vectorize tutor profiles
    vectorizer, tutor_vectors = vectorize_tutor_data(tutor_df)
    
    # Create a ChromaDB collection and add tutors
    collection = create_tutor_collection(tutor_df, tutor_vectors)
    
    # Combine student tags and description for vectorization
    student_combined = student_tags + ' ' + student_description
    student_vector = vectorizer.transform([student_combined])
    
    # Query ChromaDB for the top 5 recommended tutors
    recommended_tutors = collection.query(
        query_embeddings=student_vector.toarray().tolist(),
        n_results=5
    )
    
    # Return recommended tutors as JSON
    return JsonResponse(recommended_tutors, safe=False)
