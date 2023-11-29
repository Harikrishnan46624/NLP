from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]

# Create a CountVectorizer instance
vectorizer = CountVectorizer()

# Fit and transform the documents into a bag of words
X = vectorizer.fit_transform(documents)

# Get the feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Convert the bag of words to a dense matrix
dense_matrix = X.toarray()

# Display the result
print("Feature names (words):", feature_names)
print("Bag of words (document-term matrix):")
print(dense_matrix)
