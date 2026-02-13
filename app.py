import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# -----------------------
# Load dataset
# -----------------------
data = pd.read_csv("cleaned_ecommerce_dataset.csv")

# -----------------------
# Encode users & items
# -----------------------
user_cat = data['user_id'].astype('category')
item_cat = data['item_id'].astype('category')

data['user_code'] = user_cat.cat.codes
data['item_code'] = item_cat.cat.codes

# -----------------------
# Build sparse matrix
# -----------------------
matrix = csr_matrix(
    (data['interaction'],
     (data['user_code'], data['item_code']))
)

# -----------------------
# Train SVD model
# -----------------------
svd = TruncatedSVD(n_components=50)
user_factors = svd.fit_transform(matrix)
item_factors = svd.components_.T

# -----------------------
# Reverse maps
# -----------------------
user_map = dict(enumerate(user_cat.cat.categories))
item_map = dict(enumerate(item_cat.cat.categories))

product_to_category = data.set_index(
    'item_id')['category'].to_dict()

# -----------------------
# Recommendation function
# -----------------------
def recommend(user_id, top_n=5):

    if user_id not in user_cat.cat.categories:
        return []

    user_index = user_cat.cat.categories.get_loc(user_id)

    scores = item_factors @ user_factors[user_index]

    top_items = scores.argsort()[::-1][:top_n]

    results = []
    for idx in top_items:
        pid = item_map[idx]
        cat = product_to_category.get(pid, "Unknown")
        results.append((pid, cat, round(scores[idx],3)))

    return results

# -----------------------
# Streamlit UI
# -----------------------
st.title("🛒 E-Commerce Recommendation App")
st.write("Milestone 4 — Deployed Recommendation Tool")

uid = st.text_input("Enter User ID")
n = st.slider("Top N", 3, 10, 5)

if st.button("Recommend"):

    recs = recommend(uid, n)

    if not recs:
        st.warning("User not found")
    else:
        for pid, cat, score in recs:
            st.write(f"Product: {pid}")
            st.write(f"Category: {cat}")
            st.write(f"Score: {score}")
            st.write("---")
