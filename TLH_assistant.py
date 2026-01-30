# Copyright (c) 2025 Raul Hernandez Lopez
#
# This file is part of the project and is licensed under the
# Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0).
#
# You are free to share and adapt this file under the terms of the CC BY-SA 4.0 license.
# Full license: https://creativecommons.org/licenses/by-sa/4.0/legalcode

import streamlit as st
from rag_manager import run_rag

st.set_page_config(page_title="Asistente TLH", page_icon="📚")
st.title("Asistente de Tecnologías del Lenguaje Humano")


@st.cache_resource
def load_system():
    return run_rag()


try:
    chain = load_system()
    st.success("Sistema listo.")
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            response = chain.invoke({"query": prompt})
            result = response['result']
            sources = list(set([
                doc.metadata.get('source', 'Desconocido')
                for doc in response.get('source_documents', [])
            ]))

            full_reply = f"{result}\n\n**Fuentes:** {', '.join(sources)}"
            st.markdown(full_reply)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})