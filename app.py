# Imports

import itertools
from collections import Counter
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

import seaborn as sns
import matplotlib.pyplot as plt

import pickle

import streamlit as st

from latintools import preprocess

from tqdm import tqdm


# Constants
SOURCE = "data/lemmatized_sents.pkl"

# Cooccurence matrix helper function
def create_co_occurences_matrix(allowed_words, documents):
    # cf. https://stackoverflow.com/a/49667439
    word_to_id = dict(zip(allowed_words, range(len(allowed_words))))
    documents_as_ids = [
        np.sort([word_to_id[w] for w in doc if w in word_to_id]).astype("uint32")
        for doc in documents
    ]
    row_ind, col_ind = zip(
        *itertools.chain(
            *[[(i, w) for w in doc] for i, doc in enumerate(documents_as_ids)]
        )
    )
    data = np.ones(
        len(row_ind), dtype="uint32"
    )  # use unsigned int for better memory utilization
    max_word_id = max(itertools.chain(*documents_as_ids)) + 1
    docs_words_matrix = csr_matrix(
        (data, (row_ind, col_ind)), shape=(len(documents_as_ids), max_word_id)
    )  # efficient arithmetic operations with CSR * CSR
    words_cooc_matrix = (
        docs_words_matrix.T * docs_words_matrix
    )  # multiplying docs_words_matrix with its transpose matrix would generate the co-occurences matrix
    words_cooc_matrix.setdiag(0)
    return words_cooc_matrix, word_to_id


# Load data
@st.cache_data
def getData():
    return pickle.load(open(SOURCE, "rb")), pickle.load(open("data/terms.pkl", "rb"))


sents, terms = getData()
terms = sorted(terms)

# Make streamlit dropdown with text entry for terms

st.header("Tesserae Term Co-occurrence")
st.write(
    "Select lists of terms to create a co-occurrence matrix at the sentence level within the Tesserae Latin texts."
)

termlist = st.multiselect("Enter termlist", options=terms, key="termlist")
complist = st.multiselect("Enter termlist", options=terms + ["impurus"], key="complist")


def clear_multi():
    st.session_state.termlist = []
    st.session_state.complist = []
    return


create_plot = st.checkbox("Create plot", value=False, key="create_plot")

if st.button("Submit"):
    allowed_words = termlist + complist

    # Create matrix
    words_cooc_matrix, word_to_id = create_co_occurences_matrix(allowed_words, sents)

    # Create dataframe
    df = pd.DataFrame(
        words_cooc_matrix.todense(), index=word_to_id.keys(), columns=word_to_id.keys()
    )[complist].loc[termlist]

    st.dataframe(df)

    if create_plot:
        # Plot heatmap

        fig, ax = plt.subplots()

        sns.heatmap(df, annot=True, linewidths=0.5, square=True, cmap="Blues", ax=ax)

        plt.yticks(rotation=0)
        plt.xticks(rotation=45)

        plt.title(f"Term Co-occurrence in Tesserae", fontsize=10)
        plt.subplots_adjust(top=1.2)
        st.pyplot(fig)

if st.button("Reset", on_click=clear_multi):
    st.experimental_rerun()
