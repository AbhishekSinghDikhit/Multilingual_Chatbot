import streamlit as st
from chatbot import chatbot_response 

# Streamlit UI Configuration
st.set_page_config(page_title="Multilingual Chatbot", layout="centered")

st.title("🌍 Multilingual Chatbot with LangChain & Gemini")
st.markdown("Ask me anything in English, Hindi, Urdu, or Spanish!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history using chat_message for a polished UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input for a modern chat UX
user_input = st.chat_input("Type your message here...")

if user_input:
    # Input validation
    if user_input.strip():
        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get AI response with error handling
        try:
            with st.spinner("Thinking..."):
                response = chatbot_response(user_input)
            
            # Append AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Display AI response
            with st.chat_message("assistant"):
                st.markdown(response)
        
        except Exception as e:
            st.error(f"Oops! Something went wrong: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
    else:
        st.warning("Please enter a valid message.")

# Optional: Button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()