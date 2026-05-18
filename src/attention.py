import numpy as np

# here we have 5 tokens and each token is represented by a 4-dimensional embedding vector
tokens = ["The", "cat", "sat", "on", "the"]

# For reproducibility, we set a random seed.. it use for learning the weights of the model during training, but here we just use it to generate random embeddings for demonstration purposes.
# These values are random just for demonstration
# In real LLMs, these values are learned during training
np.random.seed(42)

# neural networks cannot work with raw text, they need to convert the words into numbers. This is done through a process called embedding, where each word is represented as a vector of numbers.
# here we use 5 tokens and each token is represented by a 4-dimensional embedding vector
# that mean each word/token is represented by a vector of 4 numbers. [0.12, 0.87, 0.33, 0.45]
embeddings = np.random.rand(5,4)

print("Embeddings:", embeddings)

# After embeddings, each word is still just a vector. The model now needs a way to transform those vectors into three different representations:
# Query (what this word is looking for)
# Key (what this word offers / represents)
# Value (the actual information to pass forward)

# To do that, we use weight matrices.
# shape is (4, 4) because we want to transform our 4-dimensional embeddings into 4-dimensional Query, Key, and Value vectors. input dimension is 4 (the size of the embedding) and output dimension is also 4 (the size of the Query, Key, and Value vectors we want to create).
# Wq - Transforms each word into a vector that represents: “What am I searching for in other words?”
# Wk - Transforms each word into a vector that represents: “What information do I contain?”
# Wv - Transforms each word into a vector that represents: “What information should I pass forward?”
Wq = np.random.rand(4, 4)  # Weight matrix for Query
Wk = np.random.rand(4, 4)  # Weight matrix for Key  
Wv = np.random.rand(4, 4)  # Weight matrix for Value

# Now we can compute the Query, Key, and Value vectors for each token by multiplying the embeddings with the respective weight matrices.
# This is done for each token, resulting in three new matrices: Q (Query), K (Key), and V (Value).
# @ is the matrix multiplication operator in Python. It multiplies the embeddings matrix (5, 4) with each of the weight matrices (4, 4) to produce new matrices of shape (5, 4) for Q, K, and V.
# After multiplication:

# Q: (5, 4)
# K: (5, 4)
# V: (5, 4)

# Meaning:

# Still 5 tokens
# But now each token has 3 different “views”
# Intuition for each token

# Take a word like "cat":

# After transformation, it becomes:

# Q_cat → what “cat” is looking for
# K_cat → how “cat” can be matched by others
# V_cat → information “cat” carries
###########################################################################################################################################################
Q = embeddings @ Wq  # Shape: (5, 4)
K = embeddings @ Wk  # Shape: (5, 4)    
V = embeddings @ Wv  # Shape: (5, 4)

# compute attention scores by taking the dot product of the Query and Key matrices. This gives us a measure of how well each token's Query matches with every other token's Key.
# This is the step where the model figures out how much each word should pay attention to every other word.
# Each token asks:
# “How relevant are the other tokens to me?”

# This is done by comparing:

# Query (Q) → what I am looking for
# Key (K) → what I represent
# The formula

# You compute similarity using a dot product:

# scores = Q Kᵀ
# Why transpose K? [one vector row should be equal to one vector column for the dot product to work]
# Q shape: (5, 4)
# K shape: (5, 4)

# To compare every token with every other token:

# Kᵀ shape = (4, 5)
# Now we can compute the attention scores by multiplying Q with the transpose of K. This will give us a matrix of shape (5, 5), where each element (i, j) represents how much token i should pay attention to token j.
scores =  Q @ K.T  # Shape: (5, 5)

# After computing raw attention scores, we need to make them numerically stable before applying softmax.
# scores = Q @ Kᵀ   → shape (5, 5)
# These values can become large when:
# embeddings grow
# model dimension increases
# dot products accumulate large numbers

# Large values cause a problem in the next step (softmax):
# softmax becomes extremely “peaky”
# gradients become unstable during training

# The solution: scaling
# Transformers fix this by dividing by: √dₖ

# Where:
# dₖ = dimension of the key vectors
# In your case: K.shape[1] = 4
# So: √dₖ = √4 = 2
dk = K.shape[1]
scaled_scores = scores / np.sqrt(dk)

# Now we can apply the softmax function to the scaled scores to get the attention weights. The softmax function converts the scores into probabilities that sum to 1, indicating how much attention each token should pay to every other token.
# The softmax function is defined as:
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True)) # Subtracting the max value for numerical stability
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True) # Normalizing to get probabilities

attention_weights = softmax(scaled_scores) # Shape: (5, 5)

# Finally, we compute the output of the attention mechanism by multiplying the attention weights with the Value matrix. This gives us a new representation for each token that incorporates information from all other tokens, weighted by their relevance.
# The output is a weighted sum of the Value vectors, where the weights are determined by how
# much attention each token pays to every other token.
output = attention_weights @ V

print("Attention Weights:")
print(attention_weights)

print("\nFinal Output:")
print(output)