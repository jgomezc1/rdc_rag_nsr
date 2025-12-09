"""
NSR-10 RAG Assistant - Streamlit Application
An AI-powered assistant for Colombian Seismic-Resistant Building Code (NSR-10)
"""

import os
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables - support both local (.env) and Streamlit Cloud (secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed on Streamlit Cloud

# Set API key from Streamlit secrets if available (for Streamlit Cloud deployment)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configuration
MODEL_NAME = "gpt-4.1-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
PERSIST_DIR = "NSR10_DB"

# Prompt templates for different modes
PROMPT_NSR10_ONLY = """
Eres un asistente experto en ingeniería estructural y en el Reglamento Colombiano de Construcción Sismo Resistente NSR-10.

Tu objetivo es guiar a ingenieros en la interpretación y aplicación de la NSR-10. Dispones de fragmentos del texto normativo bajo la sección "Contexto". Sigue estrictamente estas reglas:

1. Usa el CONTEXTO como fuente principal para toda referencia normativa.
2. Cuando cites la norma, menciona claramente el título, capítulo, artículo o numeral si esa información aparece en el contexto
   (por ejemplo: "De acuerdo con la NSR-10, Título C, Capítulo C.3, Artículo C.3.2.1, …").
3. NO inventes numerales ni texto literal de la norma que no aparezca en el contexto.
4. Si la información normativa exacta no aparece en el contexto:
   - Indica explícitamente que no se encuentra en los fragmentos proporcionados.
   - NO proporciones información adicional ni recomendaciones.

Estructura tu respuesta en UNA sección:

**Referencia normativa (NSR-10)**
- Resume el contenido relevante de la NSR-10 usando ÚNICAMENTE el contexto.
- Cita títulos, capítulos, artículos o numerales que aparezcan explícitamente en el contexto.
- Si algo no está en el contexto, dilo claramente: "No encuentro la información normativa específica en el contexto proporcionado de la NSR-10."

Contexto:
{context}

Pregunta:
{question}

Respuesta (SIEMPRE en español):
"""

PROMPT_NSR10_PLUS_RECOMMENDATIONS = """
Eres un asistente experto en ingeniería estructural y en el Reglamento Colombiano de Construcción Sismo Resistente NSR-10.

Tu objetivo es guiar a ingenieros en la interpretación y aplicación de la NSR-10. Dispones de fragmentos del texto normativo bajo la sección "Contexto". Sigue estrictamente estas reglas:

1. Usa el CONTEXTO como fuente principal para toda referencia normativa.
2. Cuando cites la norma, menciona claramente el título, capítulo, artículo o numeral si esa información aparece en el contexto
   (por ejemplo: "De acuerdo con la NSR-10, Título C, Capítulo C.3, Artículo C.3.2.1, …").
3. NO inventes numerales ni texto literal de la norma que no aparezca en el contexto.
4. Si la información normativa exacta no aparece en el contexto:
   - Indica explícitamente que no se encuentra en los fragmentos proporcionados.
   - Aún así, puedes dar una respuesta general basada en buenas prácticas de ingeniería, pero aclarando que NO es texto obligatorio de la NSR-10.

Además de citar la norma, debes complementar la respuesta con tu criterio profesional:

- Comentarios técnicos y explicaciones de comportamiento estructural.
- Buenas prácticas de diseño y de detalle constructivo.
- Recomendaciones para la práctica profesional en el contexto colombiano.

Estructura SIEMPRE tu respuesta en TRES secciones claramente separadas:

1) **Referencia normativa (NSR-10)**
   - Resume el contenido relevante de la NSR-10 usando ÚNICAMENTE el contexto.
   - Cita títulos, capítulos, artículos o numerales que aparezcan explícitamente en el contexto.
   - Si algo no está en el contexto, dilo claramente.

2) **Explicación para el ingeniero**
   - Explica con tus propias palabras qué significa esa exigencia normativa.
   - Indica por qué existe la exigencia (fundamento sismorresistente, ductilidad, redundancia, irregularidades, etc.).
   - Usa lenguaje técnico pero claro, como si hablaras con un ingeniero joven.

3) **Recomendaciones y buenas prácticas (no obligatorias)**
   - Propón criterios adicionales, verificaciones, configuraciones estructurales, detalles y chequeos que ayuden a cumplir la norma y a mejorar el desempeño.
   - Señala de forma explícita qué partes son **recomendaciones técnicas** y NO texto obligatorio de la NSR-10
     (por ejemplo: "Como recomendación técnica adicional, aunque no está explícito en la NSR-10, se sugiere…").

Si la pregunta no puede responderse con el contexto normativo dado, responde primero:

"No encuentro la información normativa específica en el contexto proporcionado de la NSR-10."

y a continuación ofrece SOLO una respuesta general basada en buenas prácticas de ingeniería estructural, sin inventar numerales ni citas específicas.

Contexto:
{context}

Pregunta:
{question}

Respuesta (SIEMPRE en español):
"""


@st.cache_resource
def load_vectordb():
    """Load the Chroma vector database from disk."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OPENAI_API_KEY not found. "
            "For local development: create a .env file with your key. "
            "For Streamlit Cloud: add OPENAI_API_KEY in Settings → Secrets."
        )
        st.stop()

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb


@st.cache_resource
def get_llm():
    """Initialize the ChatOpenAI LLM."""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=1024,
    )


def get_qa_chain(vectordb, llm, mode: str):
    """Create a conversational retrieval chain based on the selected mode."""
    # Select prompt based on mode
    if mode == "Solo NSR-10":
        template = PROMPT_NSR10_ONLY
    else:
        template = PROMPT_NSR10_PLUS_RECOMMENDATIONS

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Create memory (use session state to persist across reruns)
    if "memory" not in st.session_state or st.session_state.get("current_mode") != mode:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        st.session_state.current_mode = mode

    # Create chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def main():
    # Page configuration
    st.set_page_config(
        page_title="Supervisor Técnico R&DC",
        page_icon="company_logo.png",
        layout="wide",
    )

    # Header with logo and title
    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        st.image("company_logo.png", width=150)
    with header_col2:
        st.markdown(
            """
            <h1 style='margin-bottom: 0; padding-top: 20px;'>Supervisor Técnico R&DC</h1>
            <p style='color: gray; font-style: italic;'>Asistente de IA para el Reglamento Colombiano de Construcción Sismo Resistente</p>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Sidebar with mode selector
    with st.sidebar:
        st.header("Configuración")
        mode = st.radio(
            "Modo de respuesta:",
            options=["Solo NSR-10", "NSR-10 + Recomendaciones"],
            index=1,
            help="**Solo NSR-10**: Respuestas basadas únicamente en el texto normativo.\n\n**NSR-10 + Recomendaciones**: Incluye explicaciones técnicas y buenas prácticas."
        )

        st.divider()

        # Clear conversation button
        if st.button("Limpiar conversación", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
            )
            st.rerun()

        st.divider()
        st.markdown("### Acerca de")
        st.markdown(
            "Este asistente utiliza RAG (Retrieval-Augmented Generation) "
            "para consultar la normativa NSR-10 y proporcionar respuestas precisas."
        )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # Load resources
    vectordb = load_vectordb()
    llm = get_llm()

    # Side-by-side layout: Input (left) | Output (right)
    input_col, output_col = st.columns([1, 2])

    # LEFT COLUMN: Input area
    with input_col:
        st.markdown("### Consulta")
        query = st.text_area(
            "Escriba su pregunta sobre la NSR-10:",
            height=200,
            placeholder="Ej: ¿Cuáles son los requisitos de recubrimiento para vigas según la NSR-10?",
            key="query_input"
        )

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_btn = st.button("Enviar consulta", type="primary", use_container_width=True)
        with col_btn2:
            clear_input = st.button("Limpiar", use_container_width=True)

        if clear_input:
            st.session_state.pending_query = None
            st.rerun()

        if submit_btn and query.strip():
            st.session_state.pending_query = query.strip()
            st.rerun()

        # Show conversation history in input column
        if st.session_state.messages:
            st.markdown("---")
            st.markdown("### Historial de consultas")
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    st.markdown(f"**{i//2 + 1}.** {msg['content'][:100]}{'...' if len(msg['content']) > 100 else ''}")

    # RIGHT COLUMN: Output area
    with output_col:
        st.markdown("### Respuesta")

        # Process pending query
        if st.session_state.pending_query:
            query_to_process = st.session_state.pending_query
            st.session_state.pending_query = None

            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": query_to_process})

            with st.spinner("Consultando la normativa NSR-10..."):
                qa_chain = get_qa_chain(vectordb, llm, mode)
                result = qa_chain.invoke({"question": query_to_process})

                answer = result["answer"]
                sources = []

                # Extract source information
                for doc in result.get("source_documents", []):
                    page = doc.metadata.get("page", "N/A")
                    sources.append({
                        "page": page,
                        "content": doc.page_content
                    })

                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            st.rerun()

        # Display the latest response or placeholder
        if st.session_state.messages:
            # Find the last assistant message
            for msg in reversed(st.session_state.messages):
                if msg["role"] == "assistant":
                    # Display in a container with fixed height and scroll
                    with st.container():
                        st.markdown(msg["content"])

                        # Display sources
                        if msg.get("sources"):
                            with st.expander("Ver fuentes consultadas"):
                                for i, source in enumerate(msg["sources"], 1):
                                    st.markdown(f"**Fuente {i}** (Página {source['page']})")
                                    st.text(source["content"][:500] + "...")
                                    if i < len(msg["sources"]):
                                        st.divider()
                    break
        else:
            st.info("Escriba una consulta en el panel izquierdo para obtener información sobre la NSR-10.")


if __name__ == "__main__":
    main()
