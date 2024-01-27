import streamlit as st
import sys
import time
import openai
from config import ANYSCALE_API

st.set_page_config(page_title="ChatBot", page_icon="🤖")

ANYSCALE_ENDPOINT_TOKEN = ANYSCALE_API
sys_prompt = "You are an helpful assistant."


class OpenAIChatAgent:
    def __init__(self, model: str, system_prompt: str = ""):
        self.message_history = []
        if system_prompt:
            self.add_system_prompt(system_prompt)
        self.model = model
        self.oai_client = openai.OpenAI(
            base_url="https://api.endpoints.anyscale.com/v1",
            api_key=ANYSCALE_ENDPOINT_TOKEN
        )

    def add_system_prompt(self, prompt):
        self.message_history.append({
            'role': 'system',
            'content': prompt
        })

    def greet(self):
        return None

    def process_input(self, input: str):
        self.update_message_history(input)

        response = self.oai_client.chat.completions.create(

            model=self.model,
            messages=self.message_history,
            stream=True
        )
        words = ''
        for tok in response:
            delta = tok.choices[0].delta
            if not delta:
                self.message_history.append({
                    'role': 'assistant',
                    'content': words
                })
                break
            elif delta.content:
                words += delta.content
                yield delta.content
            else:
                continue

    def update_message_history(self, inp):
        self.message_history.append({
            'role': 'user',
            'content': inp
        })


agent = OpenAIChatAgent("mistralai/Mixtral-8x7B-Instruct-v0.1", sys_prompt)
st.title("Grievance Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your query here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = agent.process_input(prompt)
        for chunk in assistant_response:
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
