"""Graphical interface (Streamlit) for the basket recommender.

Run:  streamlit run app_gui.py
"""

import streamlit as st

from recommender import BasketRecommender, load_catalog, load_transactions
from recommender.engine import SCORING_METHODS


@st.cache_resource
def get_model(min_support: float, min_confidence: float, scoring: str) -> BasketRecommender:
    model = BasketRecommender(min_support=min_support, min_confidence=min_confidence, scoring=scoring)
    return model.fit(load_transactions(), load_catalog())


st.set_page_config(page_title="Basket Recommender", page_icon="🛒")
st.title("Basket Recommender")
st.caption("Association-rule recommendations with explanations and shopper controls.")

with st.sidebar:
    st.header("Model settings")
    min_support = st.slider("Minimum support", 0.005, 0.05, 0.01, 0.005)
    min_confidence = st.slider("Minimum confidence", 0.05, 0.8, 0.2, 0.05)
    scoring = st.selectbox("Ranking score", SCORING_METHODS, index=2)
    st.header("Your preferences")
    k = st.slider("Number of suggestions", 1, 10, 5)
    diversity = st.slider("Diversity (0 = off)", 0.0, 0.9, 0.0, 0.1)

model = get_model(min_support, min_confidence, scoring)
categories = sorted(set(model.categories.values()))

basket = st.multiselect("What's in your basket?", model.items)
avoid_cats = st.multiselect("Never suggest these categories", categories)
avoid_items = st.multiselect("Never suggest these items", [i for i in model.items if i not in basket])

recs = model.recommend(basket, k, avoid_items, avoid_cats, diversity)
st.subheader("Suggestions")
st.caption(f"{len(model.rules)} rules learned from the data.")
if not recs:
    st.info("Nothing to suggest with these filters.")
for rec in recs:
    st.markdown(f"**{rec.item}** · _{rec.category}_  \n{rec.reason}")
