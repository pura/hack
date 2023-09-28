import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained word embedding model (e.g., spaCy's medium-sized English model)
nlp = spacy.load("en_core_web_md")

offer_asos = """
ASOS Discount
Fashion Sale
30% Off Selected Orders
"""

offer_goldsmiths = """
Massimo Dutti
Mango
Zara
"""

# Sample article text
article_text = """
Fashion Tips
High Street Fashion
ASOS
H&M
Marks & Spencer
COS
ARKET
& Other Stories
Massimo Dutti
Mango
Zara
Free People
Stylish Outfits
Affordable Fashion
Expensive-Looking Pieces
Shopping Tips
Budget-Friendly Fashion
Chic Wardrobe
Trendy Accessories
Fashion Editor's Picks
"""

# Tokenize the article
tokens_articles = nlp(article_text)
# Compute the average word vector for the article
article_vector = sum(token.vector for token in tokens_articles) / len(tokens_articles)

from gensim.models import Word2Vec
sentences = [article_text.split()]
model = Word2Vec(article_text, vector_size=100, window=5, min_count=1, sg=0)
word_vector = model.wv["Chic"]

print(word_vector)

# Tokenize the article
tokens_asos = nlp(offer_asos)
# Compute the average word vector for the article
asos_vector = sum(token.vector for token in tokens_asos) / len(tokens_asos)

# Tokenize the article
tokens_gs = nlp(offer_goldsmiths)
# Compute the average word vector for the article
gs_vector = sum(token.vector for token in tokens_gs) / len(tokens_gs)

# Example vectors (replace these with your actual vectors)
# vector1 = np.array([0.2, 0.4, 0.1, 0.6, 0.3])
# vector2 = np.array([0.1, 0.5, 0.2, 0.3, 0.7])

# Reshape the vectors to have a shape of (1, n) if they are 1D arrays
vector1 = article_vector.reshape(1, -1)
vector2 = asos_vector.reshape(1, -1)
vector3 = gs_vector.reshape(1, -1)

# Calculate cosine similarity
similarity_score = cosine_similarity(vector1, vector2)[0][0]
print(f"Cosine Similarity: {similarity_score}")
similarity_score = cosine_similarity(vector1, vector3)[0][0]
print(f"Cosine Similarity: {similarity_score}")


