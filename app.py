import numpy as np
from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)
app.secret_key = "secret123"

# =========================
# USER LOGIN (BASIC)
# =========================
users = {
    "admin": "1234"
}

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("cleaned_ecommerce_dataset.csv")
data.columns = data.columns.str.strip().str.lower()

# Fix required columns
data['category'] = data['category'].astype(str).str.lower().str.strip()
data['product_name'] = data['item_id'].astype(str)

# Add image
data['image'] = data['category'].apply(
    lambda x: f"https://source.unsplash.com/300x300/?{x},product"
)

# Add price (if not present)
if 'price' not in data.columns:
    data['price'] = np.random.randint(200, 2000, size=len(data))

# =========================
# SVD MODEL
# =========================
user_cat = data['user_id'].astype('category')
item_cat = data['item_id'].astype('category')

data['u'] = user_cat.cat.codes
data['i'] = item_cat.cat.codes

matrix = csr_matrix((data['interaction'], (data['u'], data['i'])))

svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(matrix)
item_factors = svd.components_.T

# =========================
# MAPS
# =========================
item_map = dict(enumerate(item_cat.cat.categories))
image_map = data.drop_duplicates('item_id').set_index('item_id')['image'].to_dict()

# =========================
# HOME ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    if 'user' not in session:
        return redirect(url_for("login"))

    categories = sorted(data['category'].dropna().unique())
    products = []

    if request.method == "POST":
        selected_category = request.form.get("category")

        filtered_items = data[data['category'] == selected_category]['item_id'].unique()

        uid = data['user_id'].sample(1).values[0]

        if uid in user_cat.cat.categories:
            uidx = user_cat.cat.categories.get_loc(uid)

            scores = item_factors @ user_factors[uidx]
            top_items = scores.argsort()[::-1]

            count = 0
            for idx in top_items:
                pid = item_map[idx]

                if pid in filtered_items:
                    price = int(data[data['item_id'] == pid]['price'].values[0])

                    products.append({
                        "name": str(pid),
                        "image": image_map.get(pid, "https://via.placeholder.com/300"),
                        "price": price
                    })
                    count += 1

                if count == 8:
                    break

    return render_template("index.html", categories=categories, products=products)

# =========================
# LOGIN ROUTE
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for("home"))

    return render_template("login.html")

# =========================
# LOGOUT
# =========================
@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for("login"))

# =========================
# ADD TO CART
# =========================
@app.route("/add_to_cart", methods=["POST"])
def add_to_cart():
    name = request.form.get("name")
    image = request.form.get("image")
    price = request.form.get("price")

    # convert price to int
    try:
        price = int(float(price))
    except:
        price = 0

    item = {
        "name": name,
        "image": image,
        "price": price
    }

    if "cart" not in session:
        session["cart"] = []

    cart = session["cart"]
    cart.append(item)

    session["cart"] = cart

    return redirect("/cart")

# =========================
# REMOVE FROM CART
# =========================
@app.route("/remove_from_cart", methods=["POST"])
def remove_from_cart():
    index = int(request.form.get("index"))

    cart = session.get("cart", [])

    if 0 <= index < len(cart):
        cart.pop(index)

    session["cart"] = cart

    return redirect("/cart")

# =========================
# VIEW CART
# =========================
@app.route("/cart")
def cart():
    cart_items = session.get("cart", [])

    total = 0

    for item in cart_items:
        print("ITEM:", item)  # 🔍 DEBUG

        price = item.get("price")

        # handle all cases safely
        if price is None or price == "":
            price = 0
        else:
            try:
                price = int(float(price))
            except:
                price = 0

        total += price

    print("FINAL TOTAL:", total)  # 🔍 DEBUG

    return render_template("cart.html", products=cart_items, total=total)
# =========================
# CLEAR CART (TEMP FIX)
# =========================
@app.route("/clear_cart")
def clear_cart():
    session['cart'] = []
    return "Cart Cleared!"

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
