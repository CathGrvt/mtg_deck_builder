import streamlit as st

from mtg_ui_app.agentic_tab import render_agentic_tab
from mtg_ui_app.chat_tab import render_chat_tab
from mtg_ui_app.generate_tab import render_generate_tab
from mtg_ui_app.train_tab import render_train_tab


st.set_page_config(page_title="MTG Deck Builder UI", page_icon=":mage:", layout="wide")

st.title("MTG AI Deck Builder")
st.caption(
    "Local UI for deck generation, model training, agentic research, and RAG chatbot workflows."
)

(tab_generate, tab_train, tab_agentic, tab_chat) = st.tabs(
    ["Generate Deck", "Train Model", "Agentic Research", "Chatbot"]
)

with tab_generate:
    render_generate_tab()

with tab_train:
    render_train_tab()

with tab_agentic:
    render_agentic_tab()

with tab_chat:
    render_chat_tab()
