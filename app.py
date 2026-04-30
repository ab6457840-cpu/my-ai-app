import streamlit as st

st.set_page_config(page_title="Мой AI", page_icon="🤖")

st.title("🤖 Мой облачный чат")

if "chat" not in st.session_state:
    st.session_state.chat = []

def fake_model(text):
    return "Ты написал: " + text

user_input = st.text_input("Напиши сообщение:")

if st.button("Отправить"):
    if user_input:
        answer = fake_model(user_input)

        st.session_state.chat.append(("Ты", user_input))
        st.session_state.chat.append(("Бот", answer))

for role, msg in st.session_state.chat:
    if role == "Ты":
        st.markdown(f"**🧑 Ты:** {msg}")
    else:
        st.markdown(f"**🤖 Бот:** {msg}")