import streamlit as st
from model import generate

st.title("🤖 Моя нейросеть")

if "chat" not in st.session_state:
    st.session_state.chat = []

user = st.text_input("Ты:")

if st.button("Отправить"):
    if user:
        answer = generate(user)

        st.session_state.chat.append(("Ты", user))
        st.session_state.chat.append(("Бот", answer))

for r, m in st.session_state.chat:
    st.write(r + ":", m)
