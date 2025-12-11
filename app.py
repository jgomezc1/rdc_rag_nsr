"""
Advanced Normative Assistant - Streamlit Application
Asistente Normativo Avanzado - Aplicación Streamlit

AI assistant for Colombian structural codes: NSR-10 (primary) + ACI-318 (reference)
Asistente de IA para normativa estructural colombiana: NSR-10 (primaria) + ACI-318 (referencia)
"""

import os
import warnings
import streamlit as st
from typing import List
from datetime import datetime
import uuid
import json

# Suppress warnings
warnings.filterwarnings('ignore')

# Google Sheets imports for feedback
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# Load environment variables - support both local (.env) and Streamlit Cloud (secrets)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not needed on Streamlit Cloud

# Set API key from Streamlit secrets if available (for Streamlit Cloud deployment)
try:
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # No secrets.toml file, use .env instead

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field

# =============================================================================
# CONFIGURATION / CONFIGURACIÓN
# =============================================================================
MODEL_NAME = "gpt-4.1-mini"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
PERSIST_DIR = "combined_DB"  # Combined ChromaDB with NSR-10 and ACI-318

# =============================================================================
# UI TEXT DICTIONARIES / DICCIONARIOS DE TEXTO UI
# =============================================================================
UI_TEXT = {
    "es": {
        "page_title": "Asistente Normativo NSR-10 + ACI-318",
        "header_title": "Asistente Normativo Avanzado",
        "header_subtitle": "NSR-10 (normativa obligatoria) + ACI-318 (referencia técnica) | Colombia",
        "config_header": "Configuración",
        "language_label": "Idioma / Language:",
        "mode_label": "Modo de respuesta:",
        "mode_options": ["NSR-10 + ACI-318", "Solo NSR-10"],
        "mode_help": """**NSR-10 + ACI-318** (Recomendado): Respuestas estructuradas con:
1. Referencia normativa primaria (NSR-10)
2. Comparación con ACI-318
3. Recomendaciones técnicas

**Solo NSR-10**: Respuestas basadas únicamente en la normativa colombiana.""",
        "clear_btn": "Limpiar conversación",
        "about_header": "Acerca de",
        "about_text": """Este asistente normativo avanzado utiliza RAG para consultar:
- **NSR-10**: Normativa obligatoria en Colombia
- **ACI-318**: Referencia técnica comparativa

Las respuestas están estructuradas para distinguir claramente entre ambas fuentes.""",
        "query_header": "Consulta",
        "query_label": "Escriba su pregunta sobre normativa estructural:",
        "query_placeholder": "Ej: ¿Cuáles son los requisitos de recubrimiento para vigas?",
        "submit_btn": "Enviar consulta",
        "clear_input_btn": "Limpiar",
        "history_header": "Historial de consultas",
        "response_header": "Respuesta",
        "spinner_text": "Consultando normativa NSR-10 y ACI-318...",
        "sources_expander": "Ver fuentes consultadas",
        "nsr_sources_header": "📘 Fuentes NSR-10 (Normativa obligatoria)",
        "aci_sources_header": "📙 Fuentes ACI-318 (Referencia técnica)",
        "other_sources_header": "📄 Otras fuentes",
        "fragment_label": "Fragmento",
        "page_label": "Página",
        "welcome_msg": "Escriba una consulta en el panel izquierdo para obtener información de la NSR-10 y ACI-318.",
        "api_key_error": "OPENAI_API_KEY no encontrada. Para desarrollo local: cree un archivo .env con su clave. Para Streamlit Cloud: agregue OPENAI_API_KEY en Settings → Secrets.",
        "feedback_question": "¿Fue útil esta respuesta?",
        "feedback_thanks": "¡Gracias por tu feedback!",
        "feedback_placeholder": "Comentario opcional...",
        "feedback_submit": "Enviar feedback",
        "feedback_error": "Error al guardar feedback. Intenta de nuevo.",
    },
    "en": {
        "page_title": "Normative Assistant NSR-10 + ACI-318",
        "header_title": "Advanced Normative Assistant",
        "header_subtitle": "NSR-10 (mandatory code) + ACI-318 (technical reference) | Colombia",
        "config_header": "Settings",
        "language_label": "Idioma / Language:",
        "mode_label": "Response mode:",
        "mode_options": ["NSR-10 + ACI-318", "NSR-10 Only"],
        "mode_help": """**NSR-10 + ACI-318** (Recommended): Structured responses with:
1. Primary normative reference (NSR-10)
2. Comparison with ACI-318
3. Technical recommendations

**NSR-10 Only**: Responses based only on Colombian mandatory code.""",
        "clear_btn": "Clear conversation",
        "about_header": "About",
        "about_text": """This advanced normative assistant uses RAG to query:
- **NSR-10**: Mandatory code in Colombia
- **ACI-318**: Comparative technical reference

Responses are structured to clearly distinguish between both sources.""",
        "query_header": "Query",
        "query_label": "Enter your question about structural codes:",
        "query_placeholder": "E.g.: What are the cover requirements for beams?",
        "submit_btn": "Submit query",
        "clear_input_btn": "Clear",
        "history_header": "Query history",
        "response_header": "Response",
        "spinner_text": "Querying NSR-10 and ACI-318 codes...",
        "sources_expander": "View consulted sources",
        "nsr_sources_header": "📘 NSR-10 Sources (Mandatory code)",
        "aci_sources_header": "📙 ACI-318 Sources (Technical reference)",
        "other_sources_header": "📄 Other sources",
        "fragment_label": "Fragment",
        "page_label": "Page",
        "welcome_msg": "Enter a query in the left panel to get information from NSR-10 and ACI-318.",
        "api_key_error": "OPENAI_API_KEY not found. For local development: create a .env file with your key. For Streamlit Cloud: add OPENAI_API_KEY in Settings → Secrets.",
        "feedback_question": "Was this response helpful?",
        "feedback_thanks": "Thank you for your feedback!",
        "feedback_placeholder": "Optional comment...",
        "feedback_submit": "Submit feedback",
        "feedback_error": "Error saving feedback. Please try again.",
    }
}

# =============================================================================
# FEEDBACK MODULE / MÓDULO DE FEEDBACK
# =============================================================================

@st.cache_resource
def get_gspread_client():
    """
    Initialize Google Sheets client using Streamlit secrets.
    Returns None if credentials are not configured.
    """
    if not GSPREAD_AVAILABLE:
        return None

    try:
        # Try to get credentials from Streamlit secrets
        if "gcp_service_account" in st.secrets:
            credentials_dict = dict(st.secrets["gcp_service_account"])
            scopes = [
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=scopes
            )
            client = gspread.authorize(credentials)
            return client
    except Exception as e:
        st.warning(f"Feedback system not configured: {e}")
        return None

    return None


def get_feedback_sheet():
    """
    Get the feedback Google Sheet worksheet.
    Returns None if not configured.
    """
    client = get_gspread_client()
    if client is None:
        return None

    try:
        # Get spreadsheet ID from secrets
        if "feedback_sheet_id" in st.secrets:
            sheet_id = st.secrets["feedback_sheet_id"]
            spreadsheet = client.open_by_key(sheet_id)
            worksheet = spreadsheet.sheet1
            return worksheet
    except Exception as e:
        st.warning(f"Could not open feedback sheet: {e}")
        return None

    return None


def log_interaction(session_id: str, query: str, response: str,
                    sources_nsr: list, sources_aci: list,
                    mode: str, language: str) -> str:
    """
    Log an interaction to Google Sheets (without rating yet).
    Returns the row ID for later rating update.
    """
    worksheet = get_feedback_sheet()
    if worksheet is None:
        return None

    try:
        timestamp = datetime.now().isoformat()
        nsr_pages = ", ".join([str(s.get("page", "N/A")) for s in sources_nsr])
        aci_pages = ", ".join([str(s.get("page", "N/A")) for s in sources_aci])

        # Truncate response to avoid Google Sheets cell limits (50k chars)
        truncated_response = response[:5000] + "..." if len(response) > 5000 else response

        row_data = [
            timestamp,
            session_id,
            query,
            truncated_response,
            nsr_pages,
            aci_pages,
            mode,
            language,
            "",  # rating (empty initially)
            ""   # feedback_text (empty initially)
        ]

        worksheet.append_row(row_data, value_input_option="RAW")

        # Return the timestamp as a unique identifier for this row
        return timestamp

    except Exception as e:
        # Silently fail - don't break the app if feedback logging fails
        return None


def update_rating(session_id: str, timestamp: str, rating: int, feedback_text: str = "") -> bool:
    """
    Update the rating for a previously logged interaction.
    """
    worksheet = get_feedback_sheet()
    if worksheet is None:
        return False

    try:
        # Find the row with matching timestamp and session_id
        all_records = worksheet.get_all_records()

        for idx, record in enumerate(all_records):
            if record.get("timestamp") == timestamp and record.get("session_id") == session_id:
                row_num = idx + 2  # +2 because records are 0-indexed and row 1 is header

                # Update rating (column I = 9) and feedback_text (column J = 10)
                worksheet.update_cell(row_num, 9, rating)
                if feedback_text:
                    worksheet.update_cell(row_num, 10, feedback_text)
                return True

        return False

    except Exception as e:
        return False


def get_session_id() -> str:
    """
    Get or create a unique session ID for the current user session.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


# =============================================================================
# PROMPT TEMPLATES / PLANTILLAS DE PROMPT
# =============================================================================

# Spanish prompts
PROMPT_NSR10_SOLO_ES = """
Eres un asistente experto en ingeniería estructural y en el Reglamento Colombiano de Construcción Sismo Resistente NSR-10.

Tu objetivo es guiar a ingenieros en la interpretación y aplicación de la NSR-10. Dispones de fragmentos del texto normativo bajo la sección "Contexto". Los fragmentos pueden provenir de NSR-10 o ACI-318, identificados por el campo "code" en sus metadatos.

REGLAS ESTRICTAS:
1. Usa ÚNICAMENTE los fragmentos con code="NSR-10" del contexto para respuestas normativas.
2. IGNORA los fragmentos de ACI-318 en este modo.
3. Cuando cites la norma, menciona claramente el título, capítulo, artículo o numeral si aparece en el contexto.
4. NO inventes numerales ni texto literal que no aparezca en el contexto.
5. Si no hay información de NSR-10 en el contexto, responde:
   "No encuentro información normativa de la NSR-10 en el contexto proporcionado."

Estructura tu respuesta en UNA sección:

**Referencia normativa (NSR-10)**
- Resume el contenido relevante de la NSR-10 usando ÚNICAMENTE fragmentos con code="NSR-10".
- Cita títulos, capítulos, artículos o numerales que aparezcan explícitamente.

Historial de conversación:
{chat_history}

Contexto normativo:
{context}

Pregunta actual:
{question}

Respuesta (en español):
"""

PROMPT_NSR10_ACI318_ES = """
Eres un asistente experto en ingeniería estructural, especializado en:
- El Reglamento Colombiano de Construcción Sismo Resistente NSR-10 (normativa obligatoria en Colombia).
- El código ACI-318 del American Concrete Institute (referencia técnica y fuente conceptual).

Tu objetivo es guiar a ingenieros colombianos en la interpretación y aplicación de la NSR-10, usando ACI-318 como referencia comparativa que puede contener información más actualizada.

CONTEXTO IMPORTANTE:
- Los fragmentos del contexto tienen un campo "code" en sus metadatos: "NSR-10" o "ACI-318".
- La NSR-10 es la NORMATIVA OBLIGATORIA en Colombia.
- ACI-318 es una REFERENCIA TÉCNICA de comparación (no obligatoria en Colombia, pero puede tener criterios más actualizados).

MEMORIA CONVERSACIONAL - MUY IMPORTANTE:
- Revisa el HISTORIAL DE CONVERSACIÓN antes de responder.
- Si la pregunta actual es breve o hace referencia implícita a un tema anterior (ej: "¿Y para columnas?", "¿Importa el material?", "¿Qué más dice al respecto?"), DEBES interpretar la pregunta en el contexto de la conversación previa.
- Mantén coherencia con las respuestas anteriores.

REGLAS ESTRICTAS:
1. Usa el CONTEXTO NORMATIVO como fuente principal para toda referencia normativa.
2. Distingue SIEMPRE claramente entre lo que dice NSR-10 y lo que dice ACI-318.
3. Cuando cites cualquier norma, menciona el título, capítulo, artículo o sección si aparece en el contexto.
4. NO inventes numerales, secciones ni texto literal que no aparezca en el contexto.
5. Si no hay información suficiente, responde:
   "No encuentro información normativa suficiente en el contexto proporcionado de la NSR-10 y la ACI-318."

ESTRUCTURA TU RESPUESTA EN TRES SECCIONES:

**1) Referencia normativa primaria (NSR-10)**
- Usa EXCLUSIVAMENTE los fragmentos con code="NSR-10" del contexto.
- Resume lo que establece la NSR-10 sobre la consulta del usuario.
- Cita títulos, capítulos, artículos o numerales que aparezcan explícitamente.
- Si no hay fragmentos de NSR-10 relevantes, indica: "No se encontró información específica de NSR-10 en el contexto."

**2) Comparación y soporte con ACI-318**
- Usa los fragmentos con code="ACI-318" del contexto.
- Explica cómo trata ACI-318 el mismo tema:
  * ¿Es equivalente, más conservador, o más permisivo que NSR-10?
  * ¿Aporta criterios adicionales o más actualizados?
- Aclara que ACI-318 es referencia comparativa, pero NSR-10 es obligatoria en Colombia.
- Si no hay fragmentos de ACI-318 relevantes, indica: "No se encontró información de ACI-318 en el contexto."

**3) Recomendaciones y buenas prácticas (no obligatorias)**
- Propón recomendaciones para diseño, detallado, verificaciones adicionales.
- Aclara EXPLÍCITAMENTE que son **recomendaciones técnicas** y NO texto obligatorio.

Historial de conversación:
{chat_history}

Contexto normativo:
{context}

Pregunta actual:
{question}

Respuesta (en español):
"""

# English prompts
PROMPT_NSR10_SOLO_EN = """
You are an expert assistant in structural engineering and the Colombian Seismic-Resistant Construction Regulation NSR-10.

Your goal is to guide engineers in the interpretation and application of NSR-10. You have text fragments from the normative under the "Context" section. Fragments may come from NSR-10 or ACI-318, identified by the "code" field in their metadata.

STRICT RULES:
1. Use ONLY fragments with code="NSR-10" from the context for normative responses.
2. IGNORE ACI-318 fragments in this mode.
3. When citing the code, clearly mention the title, chapter, article, or section if it appears in the context.
4. DO NOT invent section numbers or literal text not present in the context.
5. If there is no NSR-10 information in the context, respond:
   "I cannot find NSR-10 normative information in the provided context."

Structure your response in ONE section:

**Normative Reference (NSR-10)**
- Summarize the relevant NSR-10 content using ONLY fragments with code="NSR-10".
- Cite titles, chapters, articles, or sections that appear explicitly.

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

PROMPT_NSR10_ACI318_EN = """
You are an expert assistant in structural engineering, specialized in:
- The Colombian Seismic-Resistant Construction Regulation NSR-10 (mandatory code in Colombia).
- The ACI-318 code from the American Concrete Institute (technical reference and conceptual source).

Your goal is to guide Colombian engineers in the interpretation and application of NSR-10, using ACI-318 as a comparative reference that may contain more updated information.

IMPORTANT CONTEXT:
- Context fragments have a "code" field in their metadata: "NSR-10" or "ACI-318".
- NSR-10 is the MANDATORY CODE in Colombia.
- ACI-318 is a TECHNICAL REFERENCE for comparison (not mandatory in Colombia, but may have more updated criteria).

CONVERSATIONAL MEMORY - VERY IMPORTANT:
- Review the CONVERSATION HISTORY before responding.
- If the current question is brief or implicitly references a previous topic (e.g., "And for columns?", "Does the material matter?", "What else does it say?"), you MUST interpret the question in the context of the previous conversation.
- Maintain coherence with previous responses.

STRICT RULES:
1. Use the NORMATIVE CONTEXT as the main source for all normative references.
2. ALWAYS clearly distinguish between what NSR-10 says and what ACI-318 says.
3. When citing any code, mention the title, chapter, article, or section if it appears in the context.
4. DO NOT invent section numbers or literal text not present in the context.
5. If there is not enough information, respond:
   "I cannot find sufficient normative information in the provided context from NSR-10 and ACI-318."

STRUCTURE YOUR RESPONSE IN THREE SECTIONS:

**1) Primary Normative Reference (NSR-10)**
- Use EXCLUSIVELY fragments with code="NSR-10" from the context.
- Summarize what NSR-10 establishes regarding the user's query.
- Cite titles, chapters, articles, or sections that appear explicitly.
- If no relevant NSR-10 fragments exist, indicate: "No specific NSR-10 information was found in the context."

**2) Comparison and Support with ACI-318**
- Use fragments with code="ACI-318" from the context.
- Explain how ACI-318 treats the same topic:
  * Is it equivalent, more conservative, or more permissive than NSR-10?
  * Does it provide additional or more updated criteria?
- Clarify that ACI-318 is a comparative reference, but NSR-10 is mandatory in Colombia.
- If no relevant ACI-318 fragments exist, indicate: "No ACI-318 information was found in the context."

**3) Recommendations and Best Practices (non-mandatory)**
- Propose recommendations for design, detailing, additional verifications.
- EXPLICITLY clarify that these are **technical recommendations** and NOT mandatory text.

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

# =============================================================================
# BALANCED RETRIEVER / RETRIEVER BALANCEADO
# =============================================================================
class BalancedCodeRetriever(BaseRetriever):
    """
    Custom retriever that fetches documents from both NSR-10 and ACI-318
    regardless of query language, ensuring balanced representation.

    Retriever personalizado que obtiene documentos de NSR-10 y ACI-318
    independientemente del idioma de la consulta, asegurando representación balanceada.
    """
    vectorstore: Chroma = Field(description="The Chroma vectorstore")
    k_per_code: int = Field(default=4, description="Number of documents to retrieve per code")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents from both codes with balanced representation.
        """
        all_docs = []

        # Retrieve from NSR-10
        try:
            nsr_docs = self.vectorstore.similarity_search(
                query,
                k=self.k_per_code,
                filter={"code": "NSR-10"}
            )
            all_docs.extend(nsr_docs)
        except Exception:
            pass

        # Retrieve from ACI-318
        try:
            aci_docs = self.vectorstore.similarity_search(
                query,
                k=self.k_per_code,
                filter={"code": "ACI-318"}
            )
            all_docs.extend(aci_docs)
        except Exception:
            pass

        return all_docs


# =============================================================================
# CORE FUNCTIONS / FUNCIONES PRINCIPALES
# =============================================================================
@st.cache_resource
def load_vectordb():
    """Load the Chroma vector database from disk."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    return vectordb


@st.cache_resource
def get_llm():
    """Initialize the ChatOpenAI LLM with low temperature for high fidelity."""
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=2048,
    )


def get_prompt_template(mode: str, language: str) -> PromptTemplate:
    """
    Get the appropriate prompt template based on mode and language.
    """
    is_nsr_only = "Solo" in mode or "Only" in mode

    if language == "es":
        template = PROMPT_NSR10_SOLO_ES if is_nsr_only else PROMPT_NSR10_ACI318_ES
    else:
        template = PROMPT_NSR10_SOLO_EN if is_nsr_only else PROMPT_NSR10_ACI318_EN

    return PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template,
    )


def format_chat_history(messages: list, language: str) -> str:
    """
    Format the chat history for inclusion in the prompt.

    Args:
        messages: List of message dictionaries with 'role' and 'content'
        language: "es" or "en"

    Returns:
        Formatted string of chat history
    """
    if not messages:
        if language == "es":
            return "(Sin historial previo)"
        return "(No previous history)"

    formatted = []
    user_label = "Usuario" if language == "es" else "User"
    assistant_label = "Asistente" if language == "es" else "Assistant"

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user":
            formatted.append(f"{user_label}: {content}")
        elif role == "assistant":
            # Truncate long responses to avoid context overflow
            truncated = content[:500] + "..." if len(content) > 500 else content
            formatted.append(f"{assistant_label}: {truncated}")

    return "\n\n".join(formatted)


def build_contextualized_query(question: str, chat_history: list, llm) -> str:
    """
    Build a contextualized search query based on chat history.
    For follow-up questions, combines context from previous exchanges.
    """
    if not chat_history:
        return question

    # Get the last user question and assistant response for context
    last_exchange = ""
    for msg in reversed(chat_history):
        if msg.get("role") == "user":
            last_exchange = msg.get("content", "")[:200]
            break

    # If current question seems like a follow-up (short or contains references)
    follow_up_indicators = ["?", "y ", "también", "además", "qué más", "importa", "aplica", "mismo"]
    is_follow_up = len(question) < 80 or any(ind in question.lower() for ind in follow_up_indicators)

    if is_follow_up and last_exchange:
        # Combine previous context with current question for better retrieval
        contextualized = f"{last_exchange} {question}"
        return contextualized

    return question


def query_with_sources(vectordb, llm, question: str, mode: str, language: str, chat_history: list) -> dict:
    """
    Query the vectorstore and generate a response with sources.

    Args:
        vectordb: Chroma vector database
        llm: Language model
        question: User's question
        mode: Response mode
        language: "es" or "en"
        chat_history: List of previous messages for conversational context

    Returns:
        dict with "answer" and "source_documents"
    """
    # Build contextualized query for better retrieval on follow-up questions
    search_query = build_contextualized_query(question, chat_history, llm)

    # Create balanced retriever
    retriever = BalancedCodeRetriever(
        vectorstore=vectordb,
        k_per_code=4  # 4 from NSR-10 + 4 from ACI-318 = 8 total
    )

    # Retrieve relevant documents using contextualized query
    source_documents = retriever.invoke(search_query)

    # Build context from documents with metadata
    context_parts = []
    for doc in source_documents:
        code = doc.metadata.get('code', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        context_parts.append(f"[code={code}, page={page}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Format chat history
    formatted_history = format_chat_history(chat_history, language)

    # Get prompt and generate response
    prompt = get_prompt_template(mode, language)
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": formatted_history
    })

    return {
        "answer": answer,
        "source_documents": source_documents
    }


# =============================================================================
# MAIN APPLICATION / APLICACIÓN PRINCIPAL
# =============================================================================
def main():
    # Initialize language in session state
    if "language" not in st.session_state:
        st.session_state.language = "es"

    # Page configuration
    st.set_page_config(
        page_title="NSR-10 + ACI-318 Assistant",
        page_icon="company_logo.png",
        layout="wide",
    )

    # Get UI text based on language
    lang = st.session_state.language
    txt = UI_TEXT[lang]

    # Header with logo and title
    header_col1, header_col2 = st.columns([1, 4])
    with header_col1:
        st.image("company_logo.png", width=150)
    with header_col2:
        st.markdown(
            f"""
            <h1 style='margin-bottom: 0; padding-top: 20px;'>{txt['header_title']}</h1>
            <p style='color: gray; font-style: italic;'>{txt['header_subtitle']}</p>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Sidebar with configuration
    with st.sidebar:
        st.header(txt['config_header'])

        # Language selector
        language_options = ["Español", "English"]
        current_lang_idx = 0 if lang == "es" else 1
        selected_lang = st.radio(
            txt['language_label'],
            options=language_options,
            index=current_lang_idx,
            horizontal=True
        )

        # Update language if changed
        new_lang = "es" if selected_lang == "Español" else "en"
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()

        st.divider()

        # Mode selector
        mode = st.radio(
            txt['mode_label'],
            options=txt['mode_options'],
            index=0,
            help=txt['mode_help']
        )

        st.divider()

        # Clear conversation button
        if st.button(txt['clear_btn'], use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

        st.divider()
        st.markdown(f"### {txt['about_header']}")
        st.markdown(txt['about_text'])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    # Load resources
    vectordb = load_vectordb()
    if vectordb is None:
        st.error(txt['api_key_error'])
        st.stop()

    llm = get_llm()

    # Two-column layout: Input (left) | Conversation (right)
    input_col, chat_col = st.columns([1, 2])

    # LEFT COLUMN: Query input area
    with input_col:
        st.markdown(f"### {txt['query_header']}")

        query = st.text_area(
            txt['query_label'],
            height=150,
            placeholder=txt['query_placeholder'],
            key="query_input"
        )

        submit_btn = st.button(txt['submit_btn'], type="primary", use_container_width=True)

        if submit_btn and query.strip():
            st.session_state.pending_query = query.strip()
            st.rerun()

        # Show conversation count
        if st.session_state.messages:
            num_exchanges = len([m for m in st.session_state.messages if m["role"] == "user"])
            if lang == "es":
                st.caption(f"💬 {num_exchanges} consulta(s) en esta conversación")
            else:
                st.caption(f"💬 {num_exchanges} query(ies) in this conversation")

    # RIGHT COLUMN: Full conversation history
    with chat_col:
        st.markdown(f"### {txt['response_header']}")

        # Container for conversation with scroll
        chat_container = st.container(height=500)

        with chat_container:
            if st.session_state.messages:
                for msg_idx, msg in enumerate(st.session_state.messages):
                    if msg["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(msg["content"])
                    elif msg["role"] == "assistant":
                        with st.chat_message("assistant"):
                            st.markdown(msg["content"])

                            # Show sources grouped by code
                            if msg.get("sources"):
                                with st.expander(txt['sources_expander']):
                                    nsr_sources = [s for s in msg["sources"] if s.get("code") == "NSR-10"]
                                    aci_sources = [s for s in msg["sources"] if s.get("code") == "ACI-318"]
                                    other_sources = [s for s in msg["sources"] if s.get("code") not in ["NSR-10", "ACI-318"]]

                                    # NSR-10 sources
                                    if nsr_sources:
                                        st.markdown(f"#### {txt['nsr_sources_header']}")
                                        for i, source in enumerate(nsr_sources, 1):
                                            st.markdown(f"**{txt['fragment_label']} {i}** - {txt['page_label']} {source['page']}")
                                            st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                                            if i < len(nsr_sources):
                                                st.divider()

                                    # ACI-318 sources
                                    if aci_sources:
                                        if nsr_sources:
                                            st.divider()
                                        st.markdown(f"#### {txt['aci_sources_header']}")
                                        for i, source in enumerate(aci_sources, 1):
                                            st.markdown(f"**{txt['fragment_label']} {i}** - {txt['page_label']} {source['page']}")
                                            st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                                            if i < len(aci_sources):
                                                st.divider()

                                    # Other sources
                                    if other_sources:
                                        if nsr_sources or aci_sources:
                                            st.divider()
                                        st.markdown(f"#### {txt['other_sources_header']}")
                                        for i, source in enumerate(other_sources, 1):
                                            st.markdown(f"**{txt['fragment_label']} {i}** ({source.get('code', 'N/A')}) - {txt['page_label']} {source['page']}")
                                            st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                                            if i < len(other_sources):
                                                st.divider()

                            # Feedback rating buttons
                            interaction_id = msg.get("interaction_id")
                            if interaction_id and not msg.get("rated", False):
                                st.caption(txt['feedback_question'])
                                col1, col2, col3 = st.columns([1, 1, 4])
                                with col1:
                                    if st.button("👍", key=f"thumbs_up_{msg_idx}", help="Útil / Helpful"):
                                        session_id = get_session_id()
                                        if update_rating(session_id, interaction_id, 5):
                                            st.session_state.messages[msg_idx]["rated"] = True
                                            st.session_state.messages[msg_idx]["rating"] = 5
                                            st.rerun()
                                with col2:
                                    if st.button("👎", key=f"thumbs_down_{msg_idx}", help="No útil / Not helpful"):
                                        session_id = get_session_id()
                                        if update_rating(session_id, interaction_id, 1):
                                            st.session_state.messages[msg_idx]["rated"] = True
                                            st.session_state.messages[msg_idx]["rating"] = 1
                                            st.rerun()
                            elif msg.get("rated", False):
                                rating = msg.get("rating", 0)
                                if rating >= 4:
                                    st.caption(f"✅ {txt['feedback_thanks']} 👍")
                                else:
                                    st.caption(f"✅ {txt['feedback_thanks']} 👎")
            else:
                st.info(txt['welcome_msg'])

    # Process pending query
    if st.session_state.pending_query:
        query_to_process = st.session_state.pending_query
        st.session_state.pending_query = None

        # Get previous conversation history (before adding current message)
        previous_messages = st.session_state.messages.copy()

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query_to_process})

        with st.spinner(txt['spinner_text']):
            # Pass previous messages for conversational context
            result = query_with_sources(vectordb, llm, query_to_process, mode, lang, previous_messages)

            answer = result["answer"]
            sources = []

            # Extract source information with code
            for doc in result.get("source_documents", []):
                page = doc.metadata.get("page", "N/A")
                code = doc.metadata.get("code", "Unknown")
                sources.append({
                    "page": page,
                    "code": code,
                    "content": doc.page_content
                })

            # Separate sources by code for logging
            nsr_sources = [s for s in sources if s.get("code") == "NSR-10"]
            aci_sources = [s for s in sources if s.get("code") == "ACI-318"]

            # Log interaction to Google Sheets
            session_id = get_session_id()
            interaction_id = log_interaction(
                session_id=session_id,
                query=query_to_process,
                response=answer,
                sources_nsr=nsr_sources,
                sources_aci=aci_sources,
                mode=mode,
                language=lang
            )

            # Add assistant message to history with interaction_id for rating
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "interaction_id": interaction_id,
                "rated": False
            })

        st.rerun()


if __name__ == "__main__":
    main()
