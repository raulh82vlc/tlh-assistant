import streamlit as st

from src.rag_system.rag_manager import run_rag


def main():
    st.set_page_config(page_title="Asistente TLH", page_icon="📚")
    st.title("Asistente de Tecnologías del Lenguaje Humano")

    @st.cache_resource
    def load_system():
        return run_rag()

    try:
        chain = load_system()
        st.success("Sistema listo.")
    except Exception as error:
        st.error(f"Error: {error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Escribe tu pregunta..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando..."):
                response = chain.invoke({"query": prompt})
                result = response["result"]
                sources = list(
                    {
                        doc.metadata.get("source", "Desconocido")
                        for doc in response.get("source_documents", [])
                    }
                )

                full_reply = f"{result}\n\n**Fuentes:** {', '.join(sources)}"
                st.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})
