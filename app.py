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
PERSIST_DIR = "chroma_db_chunkr"  # ChromaDB built from Chunkr.ai output
COLLECTION_NAME = "structural_codes_chunkr"

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
        "response_type_label": "Tipo de respuesta:",
        "response_type_options": ["Solo citas normativas", "Citas + Recomendaciones"],
        "response_type_help": """**Solo citas normativas** (Modo diccionario): Solo cita el texto exacto de la norma sin interpretaciones ni recomendaciones.

**Citas + Recomendaciones** (Recomendado): Incluye citas normativas más recomendaciones técnicas y buenas prácticas.""",
        "code_source_label": "Fuente normativa:",
        "code_source_options": ["NSR-10 + ACI-318", "Solo NSR-10", "Solo ACI-318"],
        "code_source_help": """**NSR-10 + ACI-318**: Consulta ambas normas con comparación.

**Solo NSR-10**: Solo normativa colombiana obligatoria.

**Solo ACI-318**: Solo código ACI-318 como referencia primaria.""",
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
        "feedback_submit": "Enviar",
        "feedback_skip": "Omitir",
        "feedback_comment_prompt": "¿Qué podría mejorar? (opcional)",
        "feedback_error": "Error al guardar feedback. Intenta de nuevo.",
        "expand_references_label": "Expandir referencias cruzadas",
        "expand_references_help": """**Expandir referencias cruzadas**: Cuando está activado, el sistema busca automáticamente los artículos referenciados dentro de los documentos encontrados.

Esto proporciona contexto adicional cuando un artículo cita otros artículos (ej: C.21.12.4.4 → C.21.6.4.2 → C.7.10.4), pero puede aumentar el tiempo de respuesta.""",
        "primary_source_label": "Fuente primaria",
        "referenced_source_label": "Referencia cruzada",
        "referenced_by_label": "Referenciado por",
        "expansion_stats": "referencias adicionales encontradas",
        "reference_chain_header": "Cadena de referencias",
        "reference_chain_empty": "No se encontraron referencias cruzadas",
        "primary_marker": "consulta directa",
        "advanced_options_header": "Opciones avanzadas",
    },
    "en": {
        "page_title": "Normative Assistant NSR-10 + ACI-318",
        "header_title": "Advanced Normative Assistant",
        "header_subtitle": "NSR-10 (mandatory code) + ACI-318 (technical reference) | Colombia",
        "config_header": "Settings",
        "language_label": "Idioma / Language:",
        "response_type_label": "Response type:",
        "response_type_options": ["Normative citations only", "Citations + Recommendations"],
        "response_type_help": """**Normative citations only** (Dictionary mode): Only quotes the exact normative text without interpretations or recommendations.

**Citations + Recommendations** (Recommended): Includes normative citations plus technical recommendations and best practices.""",
        "code_source_label": "Code source:",
        "code_source_options": ["NSR-10 + ACI-318", "NSR-10 Only", "ACI-318 Only", "ASCE-7 + LATBSDC + ACI-318 (PBDE)"],
        "code_source_help": """**NSR-10 + ACI-318**: Query both codes with comparison.

**NSR-10 Only**: Only Colombian mandatory code.

**ACI-318 Only**: Only ACI-318 code as primary reference.

**ASCE-7 + LATBSDC + ACI-318 (PBDE)**: Performance-Based Design bundle for US codes. Queries ASCE 7-16, LATBSDC Guidelines, and ACI-318 equally for peer review assistance.""",
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
        "asce_sources_header": "📗 ASCE-7 Sources (Load Criteria)",
        "latbsdc_sources_header": "📕 LATBSDC Sources (PBD Guidelines)",
        "other_sources_header": "📄 Other sources",
        "pbde_spinner_text": "Querying ASCE-7, LATBSDC, and ACI-318 codes...",
        "fragment_label": "Fragment",
        "page_label": "Page",
        "welcome_msg": "Enter a query in the left panel to get information from NSR-10 and ACI-318.",
        "api_key_error": "OPENAI_API_KEY not found. For local development: create a .env file with your key. For Streamlit Cloud: add OPENAI_API_KEY in Settings → Secrets.",
        "feedback_question": "Was this response helpful?",
        "feedback_thanks": "Thank you for your feedback!",
        "feedback_placeholder": "Optional comment...",
        "feedback_submit": "Submit",
        "feedback_skip": "Skip",
        "feedback_comment_prompt": "What could be improved? (optional)",
        "feedback_error": "Error saving feedback. Please try again.",
        "expand_references_label": "Expand cross-references",
        "expand_references_help": """**Expand cross-references**: When enabled, the system automatically fetches articles referenced within the retrieved documents.

This provides additional context when an article cites other articles (e.g., C.21.12.4.4 → C.21.6.4.2 → C.7.10.4), but may increase response time.""",
        "primary_source_label": "Primary source",
        "referenced_source_label": "Cross-reference",
        "referenced_by_label": "Referenced by",
        "expansion_stats": "additional references found",
        "reference_chain_header": "Reference Chain",
        "reference_chain_empty": "No cross-references found",
        "primary_marker": "direct query",
        "advanced_options_header": "Advanced Options",
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

# Dictionary mode prompts (citations only, no recommendations)
PROMPT_DICTIONARY_BOTH_ES = """
Eres un asistente que actúa como diccionario normativo de ingeniería estructural.

Tu objetivo es citar ÚNICAMENTE el texto exacto de las normas NSR-10 y ACI-318 sin interpretaciones ni recomendaciones.

CONTEXTO IMPORTANTE:
- Los fragmentos del contexto tienen un campo "code" en sus metadatos: "NSR-10" o "ACI-318".
- NSR-10 es la NORMATIVA OBLIGATORIA en Colombia.
- ACI-318 es una REFERENCIA TÉCNICA.

MEMORIA CONVERSACIONAL:
- Revisa el HISTORIAL DE CONVERSACIÓN antes de responder.
- Si la pregunta es breve o hace referencia a un tema anterior, interpreta en contexto.

REGLAS ESTRICTAS:
1. Cita ÚNICAMENTE el texto normativo exacto que aparece en el contexto.
2. NO proporciones interpretaciones, análisis ni recomendaciones.
3. NO inventes numerales ni texto que no aparezca en el contexto.
4. Distingue claramente entre lo que dice NSR-10 y lo que dice ACI-318.
5. Menciona el título, capítulo, artículo o sección si aparece en el contexto.
6. Si no hay información suficiente, responde:
   "No encuentro información normativa en el contexto proporcionado."

ESTRUCTURA TU RESPUESTA EN DOS SECCIONES:

**Referencia normativa (NSR-10)**
- Cita el texto exacto de los fragmentos con code="NSR-10".
- Indica títulos, capítulos, artículos o numerales explícitos.
- Si no hay fragmentos de NSR-10: "No se encontró información de NSR-10 en el contexto."

**Referencia normativa (ACI-318)**
- Cita el texto exacto de los fragmentos con code="ACI-318".
- Indica secciones o artículos explícitos.
- Si no hay fragmentos de ACI-318: "No se encontró información de ACI-318 en el contexto."

Historial de conversación:
{chat_history}

Contexto normativo:
{context}

Pregunta actual:
{question}

Respuesta (en español):
"""

PROMPT_DICTIONARY_BOTH_EN = """
You are an assistant acting as a normative dictionary for structural engineering.

Your goal is to quote ONLY the exact text from NSR-10 and ACI-318 codes without interpretations or recommendations.

IMPORTANT CONTEXT:
- Context fragments have a "code" field in their metadata: "NSR-10" or "ACI-318".
- NSR-10 is the MANDATORY CODE in Colombia.
- ACI-318 is a TECHNICAL REFERENCE.

CONVERSATIONAL MEMORY:
- Review the CONVERSATION HISTORY before responding.
- If the question is brief or references a previous topic, interpret in context.

STRICT RULES:
1. Quote ONLY the exact normative text that appears in the context.
2. DO NOT provide interpretations, analysis, or recommendations.
3. DO NOT invent section numbers or text not present in the context.
4. Clearly distinguish between what NSR-10 says and what ACI-318 says.
5. Mention the title, chapter, article, or section if it appears in the context.
6. If there is not enough information, respond:
   "I cannot find normative information in the provided context."

STRUCTURE YOUR RESPONSE IN TWO SECTIONS:

**Normative Reference (NSR-10)**
- Quote the exact text from fragments with code="NSR-10".
- Indicate explicit titles, chapters, articles, or sections.
- If no NSR-10 fragments: "No NSR-10 information was found in the context."

**Normative Reference (ACI-318)**
- Quote the exact text from fragments with code="ACI-318".
- Indicate explicit sections or articles.
- If no ACI-318 fragments: "No ACI-318 information was found in the context."

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

PROMPT_DICTIONARY_NSR_ES = """
Eres un asistente que actúa como diccionario normativo de la NSR-10.

Tu objetivo es citar ÚNICAMENTE el texto exacto de la NSR-10 sin interpretaciones ni recomendaciones.

REGLAS ESTRICTAS:
1. Usa ÚNICAMENTE los fragmentos con code="NSR-10" del contexto.
2. IGNORA los fragmentos de ACI-318.
3. Cita el texto normativo exacto sin interpretaciones.
4. NO proporciones recomendaciones ni análisis.
5. Menciona el título, capítulo, artículo o numeral si aparece en el contexto.
6. NO inventes numerales ni texto que no aparezca en el contexto.
7. Si no hay información de NSR-10, responde:
   "No encuentro información de la NSR-10 en el contexto proporcionado."

ESTRUCTURA TU RESPUESTA:

**Referencia normativa (NSR-10)**
- Cita el texto exacto de los fragmentos con code="NSR-10".
- Indica títulos, capítulos, artículos o numerales explícitos.

Historial de conversación:
{chat_history}

Contexto normativo:
{context}

Pregunta actual:
{question}

Respuesta (en español):
"""

PROMPT_DICTIONARY_NSR_EN = """
You are an assistant acting as a normative dictionary for NSR-10.

Your goal is to quote ONLY the exact text from NSR-10 without interpretations or recommendations.

STRICT RULES:
1. Use ONLY fragments with code="NSR-10" from the context.
2. IGNORE ACI-318 fragments.
3. Quote the exact normative text without interpretations.
4. DO NOT provide recommendations or analysis.
5. Mention the title, chapter, article, or section if it appears in the context.
6. DO NOT invent section numbers or text not present in the context.
7. If there is no NSR-10 information, respond:
   "I cannot find NSR-10 information in the provided context."

STRUCTURE YOUR RESPONSE:

**Normative Reference (NSR-10)**
- Quote the exact text from fragments with code="NSR-10".
- Indicate explicit titles, chapters, articles, or sections.

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

PROMPT_DICTIONARY_ACI_ES = """
Eres un asistente que actúa como diccionario normativo del código ACI-318.

Tu objetivo es citar ÚNICAMENTE el texto exacto del ACI-318 sin interpretaciones ni recomendaciones.

REGLAS ESTRICTAS:
1. Usa ÚNICAMENTE los fragmentos con code="ACI-318" del contexto.
2. IGNORA los fragmentos de NSR-10.
3. Cita el texto normativo exacto sin interpretaciones.
4. NO proporciones recomendaciones ni análisis.
5. Menciona la sección o artículo si aparece en el contexto.
6. NO inventes numerales ni texto que no aparezca en el contexto.
7. Si no hay información de ACI-318, responde:
   "No encuentro información del ACI-318 en el contexto proporcionado."

ESTRUCTURA TU RESPUESTA:

**Referencia normativa (ACI-318)**
- Cita el texto exacto de los fragmentos con code="ACI-318".
- Indica secciones o artículos explícitos.

Historial de conversación:
{chat_history}

Contexto normativo:
{context}

Pregunta actual:
{question}

Respuesta (en español):
"""

PROMPT_DICTIONARY_ACI_EN = """
You are an assistant acting as a normative dictionary for ACI-318 code.

Your goal is to quote ONLY the exact text from ACI-318 without interpretations or recommendations.

STRICT RULES:
1. Use ONLY fragments with code="ACI-318" from the context.
2. IGNORE NSR-10 fragments.
3. Quote the exact normative text without interpretations.
4. DO NOT provide recommendations or analysis.
5. Mention the section or article if it appears in the context.
6. DO NOT invent section numbers or text not present in the context.
7. If there is no ACI-318 information, respond:
   "I cannot find ACI-318 information in the provided context."

STRUCTURE YOUR RESPONSE:

**Normative Reference (ACI-318)**
- Quote the exact text from fragments with code="ACI-318".
- Indicate explicit sections or articles.

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

# ACI-318 Solo prompts (with recommendations, treating ACI as primary)
PROMPT_ACI318_SOLO_ES = """
Eres un asistente experto en ingeniería estructural y en el código ACI-318 del American Concrete Institute.

Tu objetivo es guiar a ingenieros en la interpretación y aplicación del ACI-318. Dispones de fragmentos del texto normativo bajo la sección "Contexto". Los fragmentos pueden provenir de NSR-10 o ACI-318, identificados por el campo "code" en sus metadatos.

REGLAS ESTRICTAS:
1. Usa ÚNICAMENTE los fragmentos con code="ACI-318" del contexto para respuestas normativas.
2. IGNORA los fragmentos de NSR-10 en este modo.
3. Cuando cites el código, menciona claramente la sección o artículo si aparece en el contexto.
4. NO inventes numerales ni texto literal que no aparezca en el contexto.
5. Si no hay información de ACI-318 en el contexto, responde:
   "No encuentro información normativa del ACI-318 en el contexto proporcionado."

ESTRUCTURA TU RESPUESTA EN DOS SECCIONES:

**1) Referencia normativa (ACI-318)**
- Resume el contenido relevante del ACI-318 usando ÚNICAMENTE fragmentos con code="ACI-318".
- Cita secciones o artículos que aparezcan explícitamente.

**2) Recomendaciones y buenas prácticas (no obligatorias)**
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

PROMPT_ACI318_SOLO_EN = """
You are an expert assistant in structural engineering and the ACI-318 code from the American Concrete Institute.

Your goal is to guide engineers in the interpretation and application of ACI-318. You have text fragments from the normative under the "Context" section. Fragments may come from NSR-10 or ACI-318, identified by the "code" field in their metadata.

STRICT RULES:
1. Use ONLY fragments with code="ACI-318" from the context for normative responses.
2. IGNORE NSR-10 fragments in this mode.
3. When citing the code, clearly mention the section or article if it appears in the context.
4. DO NOT invent section numbers or literal text not present in the context.
5. If there is no ACI-318 information in the context, respond:
   "I cannot find ACI-318 normative information in the provided context."

STRUCTURE YOUR RESPONSE IN TWO SECTIONS:

**1) Normative Reference (ACI-318)**
- Summarize the relevant ACI-318 content using ONLY fragments with code="ACI-318".
- Cite sections or articles that appear explicitly.

**2) Recommendations and Best Practices (non-mandatory)**
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
# PBDE PROMPT TEMPLATES (ASCE-7 + LATBSDC + ACI-318) - English Only
# =============================================================================

PROMPT_PBDE_RECOMMENDATIONS_EN = """
You are an expert assistant in structural engineering for performance-based design, specialized in:
- ASCE 7-16: Minimum Design Loads and Associated Criteria for Buildings
- LATBSDC: Los Angeles Tall Buildings Structural Design Council Guidelines
- ACI-318: American Concrete Institute Building Code

Your goal is to assist structural engineers in peer review of performance-based designs by providing comprehensive information from all three codes. All three codes are treated as equally important references.

IMPORTANT CONTEXT:
- Context fragments have a "code" field in their metadata: "ASCE-7", "LATBSDC", or "ACI-318".
- All three codes are authoritative references for performance-based design review.
- ASCE-7 provides load criteria and seismic design requirements.
- LATBSDC provides guidelines for tall building performance-based design.
- ACI-318 provides concrete structural requirements.

CONVERSATIONAL MEMORY:
- Review the CONVERSATION HISTORY before responding.
- If the current question references a previous topic, interpret in context.

STRICT RULES:
1. Use the NORMATIVE CONTEXT as the main source for all references.
2. ALWAYS clearly distinguish between what each code says (ASCE-7, LATBSDC, ACI-318).
3. When citing any code, mention the section or article if it appears in the context.
4. DO NOT invent section numbers or text not present in the context.
5. If there is not enough information, respond:
   "I cannot find sufficient information in the provided context from ASCE-7, LATBSDC, and ACI-318."

STRUCTURE YOUR RESPONSE IN FOUR SECTIONS:

**1) ASCE-7 Reference**
- Use fragments with code="ASCE-7" from the context.
- Summarize load criteria, seismic parameters, or relevant requirements.
- If no relevant ASCE-7 fragments: "No specific ASCE-7 information was found in the context."

**2) LATBSDC Guidelines Reference**
- Use fragments with code="LATBSDC" from the context.
- Summarize performance-based design requirements or acceptance criteria.
- If no relevant LATBSDC fragments: "No specific LATBSDC information was found in the context."

**3) ACI-318 Reference**
- Use fragments with code="ACI-318" from the context.
- Summarize structural concrete requirements.
- If no relevant ACI-318 fragments: "No specific ACI-318 information was found in the context."

**4) Integration and Recommendations**
- Explain how the three codes interact for the topic.
- Highlight any differences or complementary requirements.
- Provide recommendations for peer review considerations.
- EXPLICITLY clarify that these are **technical recommendations**.

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

PROMPT_PBDE_DICTIONARY_EN = """
You are an assistant acting as a normative dictionary for performance-based structural design codes.

Your goal is to quote ONLY the exact text from ASCE-7, LATBSDC, and ACI-318 without interpretations or recommendations.

IMPORTANT CONTEXT:
- Context fragments have a "code" field: "ASCE-7", "LATBSDC", or "ACI-318".
- All three codes are treated as equal references.

STRICT RULES:
1. Quote ONLY the exact normative text that appears in the context.
2. DO NOT provide interpretations, analysis, or recommendations.
3. DO NOT invent section numbers or text not present in the context.
4. Clearly distinguish between what each code says.
5. Mention the section or article if it appears in the context.
6. If there is not enough information, respond:
   "I cannot find normative information in the provided context."

STRUCTURE YOUR RESPONSE IN THREE SECTIONS:

**Normative Reference (ASCE-7)**
- Quote the exact text from fragments with code="ASCE-7".
- If no ASCE-7 fragments: "No ASCE-7 information was found in the context."

**Normative Reference (LATBSDC)**
- Quote the exact text from fragments with code="LATBSDC".
- If no LATBSDC fragments: "No LATBSDC information was found in the context."

**Normative Reference (ACI-318)**
- Quote the exact text from fragments with code="ACI-318".
- If no ACI-318 fragments: "No ACI-318 information was found in the context."

Conversation history:
{chat_history}

Normative context:
{context}

Current question:
{question}

Response (in English):
"""

# =============================================================================
# LATEX FORMATTING HELPER
# =============================================================================

def convert_latex_delimiters(text: str) -> str:
    r"""
    Convert LaTeX delimiters to Streamlit-compatible format.
    Streamlit markdown uses $ for inline and $$ for display math.

    Converts:
    - \[...\] -> $$...$$  (display math)
    - \(...\) -> $...$    (inline math)
    """
    import re

    # Display math: \[...\] -> $$...$$
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)

    # Inline math: \(...\) -> $...$
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)

    return text


def normalize_latex_output(text: str) -> str:
    r"""
    Comprehensive post-processing of LLM output to ensure proper LaTeX rendering.

    Fixes:
    1. Converts remaining \[...\] and \(...\) to $$ and $
    2. Wraps orphaned LaTeX commands in $ delimiters
    3. Fixes broken/mixed delimiters like $$...$ or $...$$
    4. Wraps structural engineering formulas (Vn, Vc, f'c, etc.)
    """
    import re

    # Step 1: Convert any remaining \[...\] or \(...\) delimiters
    text = convert_latex_delimiters(text)

    # Step 2: Fix broken/malformed delimiters from source data
    # These patterns handle common issues in NSR-10 source chunks
    # ORDER IS IMPORTANT: Fix unclosed $ first, then fix mixed $$/$

    # 2a. Fix unclosed $ at start of line followed by LaTeX content and newline
    # Pattern: line starts with $ + space + LaTeX content (with backslash) + newline
    # This is typically display math that should be $$...$$
    # MUST RUN FIRST to prevent 2c/2d from matching across multiple equations
    text = re.sub(
        r'^(\$)\s+([^\n$]*\\[a-zA-Z][^\n$]*?)\s*$',
        r'$$\2$$',
        text,
        flags=re.MULTILINE
    )

    # 2b. Fix unclosed $ at end of text with LaTeX content
    # Pattern: $ + space + LaTeX content (with backslash) at very end
    text = re.sub(
        r'\$\s+([^$]*\\[a-zA-Z][^$]*?)\s*$',
        r'$$\1$$',
        text
    )

    # 2c. Fix $$ followed by content and single $ (should be $$...$$)
    # Use non-greedy matching to handle multiple equations in same text
    # IMPORTANT: Require content to start with a math-like character (backslash, letter,
    # number, or opening bracket) to avoid matching closing $$ of one equation with
    # opening $ of the next (e.g., "$$eq1$$, donde $eq2$" shouldn't match "$$, donde $")
    text = re.sub(r'\$\$([\\a-zA-Z0-9({\[+\-][^$]*?)\$(?!\$)', r'$$\1$$', text)

    # 2d. Fix single $ followed by content and $$ (should be $...$)
    # Use non-greedy matching and require content without whitespace to avoid
    # matching across separate equations like "$V$: $$equation$$"
    text = re.sub(r'(?<!\$)\$([^$\s]+?)\$\$', r'$\1$', text)

    # 2e. Fix display math written as $ content $ on its own line (should be $$...$$)
    # This catches cases where someone used $ instead of $$ for display math
    text = re.sub(
        r'^(\$)\s+([^\n$]+?)\s+\$\s*$',
        r'$$\2$$',
        text,
        flags=re.MULTILINE
    )

    # Step 3: Find and wrap orphaned LaTeX expressions
    # These are LaTeX commands that appear outside of $ delimiters

    # First, mark all properly delimited sections
    placeholder_counter = [0]
    protected = {}

    def protect_delimited(match):
        key = f"__PROTECTED_{placeholder_counter[0]}__"
        placeholder_counter[0] += 1
        protected[key] = match.group(0)
        return key

    # Protect $$...$$ and $...$
    # Use non-greedy matching and DOTALL to handle multiline content
    # Also handle edge case where content might have \$ escapes
    text = re.sub(r'\$\$(?:[^$]|\$(?!\$))+?\$\$', protect_delimited, text, flags=re.DOTALL)
    text = re.sub(r'(?<!\$)\$(?!\$)(?:[^$])+?\$(?!\$)', protect_delimited, text, flags=re.DOTALL)

    # NOTE: We intentionally do NOT try to wrap complex LaTeX commands like \frac, \sqrt, etc.
    # These often have nested braces that simple regex cannot handle correctly.
    # Complex LaTeX from source documents should already have proper delimiters.
    # We only wrap simple, standalone structural engineering patterns (Step 4).

    # Step 4: Structural engineering specific patterns
    # These patterns catch common formulas that LLM may write without LaTeX delimiters
    # IMPORTANT: These only match OUTSIDE of LaTeX expressions (no backslash nearby)
    # ORDER MATTERS: More specific/longer patterns must come before simpler ones

    # Helper function to check if a match is inside LaTeX (has \ nearby)
    def is_inside_latex(text, match_start, match_end, window=20):
        """Check if match position is likely inside a LaTeX expression."""
        start = max(0, match_start - window)
        end = min(len(text), match_end + window)
        context = text[start:end]
        # If there's a backslash command nearby, it's probably LaTeX
        return '\\' in context

    # 4a. Chained comparisons with Greek letters and structural variables (MUST BE FIRST)
    # Matches: ρ_min ≤ ρ ≤ ρ_max, ρ_min ≤ ρ, ρ ≤ ρ_max
    # Only match if not inside LaTeX
    chained_comparison_pattern = (
        r'(?<!\$)'
        r'('
        r'[ρφλ]_?(?:min|max|b|s|bal)?'  # First term
        r'\s*[≤≥<>]\s*'
        r'[ρφλ]'  # Middle term
        r'(?:\s*[≤≥<>]\s*[ρφλ]_?(?:min|max|b|s|bal)?)?'  # Optional third term
        r')'
        r'(?!\$)'
    )

    def wrap_if_not_latex(pattern, text):
        """Wrap matches only if they're not inside LaTeX expressions."""
        result = []
        last_end = 0
        for match in re.finditer(pattern, text):
            result.append(text[last_end:match.start()])
            if is_inside_latex(text, match.start(), match.end()):
                result.append(match.group(0))  # Keep as-is
            else:
                result.append(f'${match.group(1)}$')  # Wrap
            last_end = match.end()
        result.append(text[last_end:])
        return ''.join(result)

    text = wrap_if_not_latex(chained_comparison_pattern, text)

    # Re-protect newly created $...$ blocks to prevent subsequent patterns from breaking them
    text = re.sub(r'(?<!\$)\$(?!\$)(?:[^$])+?\$(?!\$)', protect_delimited, text, flags=re.DOTALL)

    # 4b. Equations with structural engineering variables
    # Pattern: Variable = Expression (e.g., "Vn = Vc + Vs", "V_n = V_c + V_s")
    # Only matches simple standalone equations, not inside LaTeX
    structural_equation_pattern = (
        r'(?<!\$)'  # Not already in LaTeX
        r'(?<!\\)'  # Not preceded by backslash
        r'('
        r'[φϕ]?'  # Optional phi symbol
        r'(?:[VMPNTCA]_?[nuscrwyeogpbj]|f\'_?c|f_?[yrc]|A_?[svgctp]|b_?w|ρ_?(?:min|max|b|s)?|λ)'  # Structural variables (simplified - no braces)
        r'\s*'
        r'[=≥≤><±+\-]'  # Operator
        r'\s*'
        r'[φϕ]?'
        r'(?:[VMPNTCA]_?[nuscrwyeogpbj]|f\'_?c|f_?[yrc]|A_?[svgctp]|b_?w|ρ_?(?:min|max|b|s)?|λ|[\d.,]+)'  # Right side
        r'(?:\s*[+\-×·*/]\s*[φϕ]?(?:[VMPNTCA]_?[nuscrwyeogpbj]|f\'_?c|f_?[yrc]|A_?[svgctp]|b_?w|ρ_?(?:min|max|b|s)?|λ|[\d.,]+))*'  # Additional terms
        r')'
        r'(?!\$)'  # Not already in LaTeX
        r'(?!\\)'  # Not followed by backslash
    )
    text = wrap_if_not_latex(structural_equation_pattern, text)

    # 4c. Standalone structural variables with subscripts (not in equations)
    # Matches: V_n, V_c, V_s, M_n, M_u, A_s, A_v, f'_c, φV_n, etc.
    # Only simple underscore subscripts, no braced subscripts (those are likely inside LaTeX)
    structural_var_pattern = (
        r'(?<![\\$\w{])'  # Not preceded by backslash, dollar, word char, or brace
        r'('
        r'[φϕ]?'  # Optional phi
        r'(?:'
        r'[VMPATNC]_[nuscrwyeogpbj](?:_?[0-9])?|'  # Variables with simple subscripts: V_n, M_u, V_s1
        r'f\'_?c|'  # Concrete strength: f'c, f'_c
        r'f_[yrc]|'  # Steel/material strength: f_y, f_r, f_c
        r'A_[svgtcp]|'  # Areas: A_s, A_v, A_g
        r'b_w|'  # Width: b_w
        r'ρ_(?:min|max|b|s|bal)|'  # Reinforcement ratios: ρ_min, ρ_max
        r'l_[dn]'  # Development length: l_d, l_n
        r')'
        r')'
        r'(?![\\$\w}])'  # Not followed by backslash, dollar, word char, or brace
    )
    text = wrap_if_not_latex(structural_var_pattern, text)

    # 4d. Concrete strength notation: f'c without subscript formatting
    # Matches: f'c, √f'c, but not already in $ delimiters
    text = re.sub(
        r"(?<![\\$\w])(f'c)(?![\\$\w])",
        r"$\1$",
        text
    )

    # 4e. Expressions with square root symbol
    # √f'c, √(f'c), √f'_c, etc.
    text = re.sub(
        r'(?<!\$)(√\(?f\'_?c\)?)',
        r'$\1$',
        text
    )
    # Also handle cases where f'c was already wrapped: √$f'c$ -> $√f'c$
    text = re.sub(
        r'√\$([^$]+)\$',
        r'$√\1$',
        text
    )

    # 4f. Greek letters commonly used in structural engineering (standalone)
    # φ (phi - strength reduction), ρ (rho - reinforcement ratio), λ (lambda - lightweight factor)
    # Only match if not already in a $ context and not followed by underscore (would be variable)
    text = re.sub(
        r'(?<![\\$\w])([φϕρλ])(?![\\$\w_≤≥<>])',
        r'$\1$',
        text
    )

    # Restore protected sections
    for key, value in protected.items():
        text = text.replace(key, value)

    # Step 5: Clean up any double delimiters ($$$$) that might have been created
    text = re.sub(r'\${3,}', '$$', text)

    # Step 6: Fix adjacent $ delimiters that should be merged
    # Note: Do NOT remove $$ (display math) - only remove truly empty pairs like $ $
    text = re.sub(r'\$\s+\$', '', text)  # Remove empty $ $ pairs (with whitespace between)
    text = re.sub(r'\$([^$]+)\$\s+\$([^$]+)\$', r'$\1 \2$', text)  # Merge adjacent inline (with space between)

    # Step 7: Ensure $$ blocks are on their own lines for better rendering
    # NOTE: Disabled because the patterns can incorrectly match across multiple
    # $$...$$ blocks in the same text, treating closing $$ and opening $$ of
    # adjacent equations as a single block. Streamlit renders $$...$$ inline fine.
    # text = re.sub(r'([^\n])\$\$([^$]+)\$\$', r'\1\n$$\2$$', text)
    # text = re.sub(r'\$\$([^$]+)\$\$([^\n])', r'$$\1$$\n\2', text)

    # Step 8: SAFETY FALLBACK - Remove any remaining __PROTECTED_X__ placeholders
    # This handles edge cases where restoration failed or source data had issues
    # Use iterative approach to handle any nested placeholders
    max_iterations = 10  # Prevent infinite loops
    for _ in range(max_iterations):
        # Check if any placeholders remain
        if '__PROTECTED_' not in text:
            break
        # Try to restore from dictionary
        for key, value in protected.items():
            text = text.replace(key, value)
        # If placeholders still remain after restoration, remove them
        if '__PROTECTED_' in text:
            text = re.sub(r'__PROTECTED_\d+__', '', text)

    return text


# =============================================================================
# BALANCED RETRIEVER / RETRIEVER BALANCEADO
# =============================================================================

def extract_article_references(query: str) -> List[str]:
    """
    Extract article references from query (e.g., C.21.12.4.4, A.3.2, 25.4.2.1).

    Returns list of article patterns to search for.
    """
    import re

    # Patterns for NSR-10 articles (e.g., C.21.12.4.4, A.3.2.1, G.1.2)
    # and ACI-318 sections (e.g., 25.4.2.1, 18.7.5.2)
    patterns = [
        r'[A-Z]\.\d+(?:\.\d+)+',  # NSR-10 style: C.21.12.4.4, A.3.2.1
        r'\d+\.\d+(?:\.\d+)+',     # ACI-318 style: 25.4.2.1, 18.7.5.2
    ]

    references = []
    for pattern in patterns:
        matches = re.findall(pattern, query)
        references.extend(matches)

    return list(set(references))


def extract_section_headers(content: str) -> List[str]:
    """
    Extract section headers from document content.

    Looks for patterns like:
    - # 11.7.3 Section title (markdown headers)
    - C.21.12.4.4 (NSR-10 article references)
    - 25.4.2.1 (ACI-318 section references)

    Returns list of unique section/article numbers found.
    """
    import re

    sections = []

    # Pattern for markdown headers with section numbers (e.g., "# 11.7.3 Title")
    header_pattern = r'#\s*(\d+\.\d+(?:\.\d+)*)'
    sections.extend(re.findall(header_pattern, content))

    # Pattern for NSR-10 articles at start of line or after whitespace (e.g., "C.21.12.4.4")
    nsr_pattern = r'(?:^|\s)([A-Z]\.\d+(?:\.\d+)+)'
    sections.extend(re.findall(nsr_pattern, content))

    # Pattern for ACI-318 sections at start of line (e.g., "25.4.2.1 Text")
    aci_pattern = r'(?:^|\n)(\d+\.\d+(?:\.\d+)+)\s+[A-Z]'
    sections.extend(re.findall(aci_pattern, content))

    # Remove duplicates and sort
    # Use tuple of (is_alpha, value) to ensure consistent ordering between letters and numbers
    def section_sort_key(x):
        parts = []
        for p in re.split(r'[.\s]', x):
            if p.isdigit():
                parts.append((0, int(p)))  # Numbers sort first, by value
            elif p:
                parts.append((1, p))  # Letters sort after numbers, alphabetically
        return parts

    try:
        unique_sections = sorted(set(sections), key=section_sort_key)
    except Exception:
        # Fallback to simple alphabetical sort if comparison fails
        unique_sections = sorted(set(sections))

    # Return first 3 sections to keep it compact
    return unique_sections[:3]


def build_reference_chain_tree(sources: List[dict], txt: dict) -> str:
    """
    Build a visual tree representation of the reference chain.

    Args:
        sources: List of source dictionaries with keys: page, code, sections, source_type, referenced_by
        txt: UI text dictionary for localization

    Returns:
        Formatted string with Unicode tree characters showing the reference chain
    """
    if not sources:
        return ""

    # Separate by code
    nsr_sources = [s for s in sources if s.get("code") == "NSR-10"]
    aci_sources = [s for s in sources if s.get("code") == "ACI-318"]

    # Check if there are any referenced sources (cross-references were expanded)
    has_references = any(s.get("source_type") == "referenced" for s in sources)
    if not has_references:
        return ""  # No chain to display if no cross-references

    lines = []

    def build_tree_for_code(code_sources: List[dict], code_name: str):
        """Build tree for a single code (NSR-10 or ACI-318)."""
        if not code_sources:
            return []

        tree_lines = []
        tree_lines.append(f"**{code_name}**")

        # Separate primary and referenced
        primary = [s for s in code_sources if s.get("source_type", "primary") == "primary"]
        referenced = [s for s in code_sources if s.get("source_type") == "referenced"]

        if not primary and not referenced:
            return []

        # Build a map: section -> source info
        section_to_source = {}
        for s in code_sources:
            sections = s.get("sections", [])
            key = sections[0] if sections else f"Pág.{s.get('page', '?')}"
            section_to_source[key] = s

        # Build adjacency: referenced_by -> list of sections that reference it
        # This creates child -> parent relationships
        children_of = {}  # parent_section -> [child_sections]
        root_sections = []  # sections with no parent (primary sources)

        for s in code_sources:
            sections = s.get("sections", [])
            section_key = sections[0] if sections else f"Pág.{s.get('page', '?')}"
            referenced_by = s.get("referenced_by", [])

            if s.get("source_type") == "primary" or not referenced_by:
                root_sections.append(section_key)
            else:
                # This source was referenced by other sections
                for parent in referenced_by:
                    if parent not in children_of:
                        children_of[parent] = []
                    if section_key not in children_of[parent]:
                        children_of[parent].append(section_key)

        # Remove duplicates from root
        root_sections = list(dict.fromkeys(root_sections))

        # Track visited sections to prevent circular reference loops
        visited = set()

        # Recursive function to build tree
        def add_branch(section_key, prefix="", is_last=True):
            # Prevent infinite recursion from circular references
            if section_key in visited:
                return []
            visited.add(section_key)

            branch_lines = []
            source = section_to_source.get(section_key)

            # Determine connector
            connector = "└─" if is_last else "├─"

            # Build the line
            if source:
                page = source.get("page", "?")
                is_primary = source.get("source_type", "primary") == "primary"
                marker = f" ← *{txt.get('primary_marker', 'direct query')}*" if is_primary else ""
                branch_lines.append(f"{prefix}{connector} §{section_key} (Pág. {page}){marker}")
            else:
                # Section referenced but not found in our sources
                branch_lines.append(f"{prefix}{connector} §{section_key}")

            # Add children
            children = children_of.get(section_key, [])
            child_prefix = prefix + ("   " if is_last else "│  ")
            for i, child in enumerate(children):
                child_is_last = (i == len(children) - 1)
                branch_lines.extend(add_branch(child, child_prefix, child_is_last))

            return branch_lines

        # Build tree from roots
        for i, root in enumerate(root_sections):
            is_last_root = (i == len(root_sections) - 1)
            tree_lines.extend(add_branch(root, "", is_last_root))

        return tree_lines

    # Build trees for each code
    nsr_tree = build_tree_for_code(nsr_sources, "NSR-10")
    aci_tree = build_tree_for_code(aci_sources, "ACI-318")

    if nsr_tree:
        lines.extend(nsr_tree)
    if aci_tree:
        if nsr_tree:
            lines.append("")  # Spacing between codes
        lines.extend(aci_tree)

    return "\n".join(lines) if lines else ""


def categorize_reference_code(reference: str, source_code: str = None) -> str:
    """
    Determine which code a reference belongs to based on its pattern.

    Args:
        reference: Article reference string (e.g., "C.21.12.4.4" or "25.4.2.1")
        source_code: Code of the document where reference was found (fallback)

    Returns:
        "NSR-10" or "ACI-318"
    """
    import re
    # NSR-10 pattern: starts with letter (C.21.12.4.4, A.3.2.1)
    if re.match(r'^[A-Z]\.', reference):
        return "NSR-10"
    # ACI-318 pattern: starts with number (25.4.2.1)
    elif re.match(r'^\d+\.', reference):
        return "ACI-318"
    # Fallback to source code if ambiguous
    return source_code or "NSR-10"


def keyword_search_documents(vectorstore, references: List[str], code_filter: str) -> List[Document]:
    """
    Search for documents containing specific article references.
    This is a direct text search, not semantic search.
    """
    if not references:
        return []

    found_docs = []

    try:
        # Get the collection for direct access
        collection = vectorstore._collection
        all_data = collection.get(include=["documents", "metadatas"])

        # Determine which codes to include based on filter
        if code_filter == "pbde":
            # PBDE bundle: ASCE-7 + LATBSDC + ACI-318
            allowed_codes = {"ASCE-7", "LATBSDC", "ACI-318"}
        elif code_filter == "both":
            # Colombian bundle: NSR-10 + ACI-318
            allowed_codes = {"NSR-10", "ACI-318"}
        elif code_filter == "nsr":
            allowed_codes = {"NSR-10"}
        elif code_filter == "aci":
            allowed_codes = {"ACI-318"}
        else:
            allowed_codes = {"NSR-10", "ACI-318"}

        for doc_content, metadata in zip(all_data['documents'], all_data['metadatas']):
            code = metadata.get('code', '')

            # Check code filter
            if code not in allowed_codes:
                continue

            # Check if any reference is in this document
            for ref in references:
                if ref in doc_content:
                    # Create a Document object
                    found_docs.append(Document(
                        page_content=doc_content,
                        metadata=metadata
                    ))
                    break  # Don't add same doc multiple times

    except Exception:
        pass

    return found_docs


def resolve_references_recursively(
    vectorstore,
    initial_docs: List[Document],
    max_articles: int = 15
) -> tuple:
    """
    Recursively resolve article references within documents.
    Only follows references within the same code (NSR->NSR, ACI->ACI).

    Args:
        vectorstore: Chroma vector database
        initial_docs: Primary retrieved documents
        max_articles: Maximum total articles to fetch (cap)

    Returns:
        Tuple of:
        - all_docs: List of all documents (primary + referenced)
        - reference_chain: Dict mapping article refs to their referencing articles
    """
    from collections import deque

    all_docs = list(initial_docs)
    reference_chain = {}  # {referenced_article: [list of articles that reference it]}

    # Track visited articles by their references found in content
    visited_refs = set()

    # Track seen pages to avoid duplicate documents
    seen_pages = set()
    for doc in initial_docs:
        page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
        seen_pages.add(page_key)

    # Extract initial references from primary documents
    pending_refs = deque()  # (reference, source_article, source_code)

    for doc in initial_docs:
        doc_code = doc.metadata.get('code', '')
        refs_in_doc = extract_article_references(doc.page_content)
        for ref in refs_in_doc:
            ref_code = categorize_reference_code(ref, doc_code)
            # Only follow same-code references (NSR->NSR, ACI->ACI)
            if ref_code == doc_code and ref not in visited_refs:
                # Find a reference from the source doc to use as "referenced by"
                source_refs = extract_article_references(doc.page_content)
                source_ref = source_refs[0] if source_refs else f"Page {doc.metadata.get('page', 'N/A')}"
                pending_refs.append((ref, source_ref, doc_code))

    # BFS to resolve references
    while pending_refs and len(all_docs) < max_articles:
        ref, source_ref, target_code = pending_refs.popleft()

        if ref in visited_refs:
            continue

        visited_refs.add(ref)

        # Search for documents containing this reference
        code_filter = "nsr" if target_code == "NSR-10" else "aci"
        found_docs = keyword_search_documents(vectorstore, [ref], code_filter)

        for doc in found_docs:
            if len(all_docs) >= max_articles:
                break

            page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
            if page_key in seen_pages:
                continue

            seen_pages.add(page_key)

            # Track reference chain
            if ref not in reference_chain:
                reference_chain[ref] = []
            if source_ref not in reference_chain[ref]:
                reference_chain[ref].append(source_ref)

            # Mark as referenced document
            doc.metadata['source_type'] = 'referenced'
            doc.metadata['referenced_by'] = reference_chain[ref].copy()
            all_docs.append(doc)

            # Extract new references from this document for further expansion
            doc_code = doc.metadata.get('code', '')
            new_refs = extract_article_references(doc.page_content)
            for new_ref in new_refs:
                new_ref_code = categorize_reference_code(new_ref, doc_code)
                # Only follow same-code references
                if new_ref_code == doc_code and new_ref not in visited_refs:
                    pending_refs.append((new_ref, ref, doc_code))

    return all_docs, reference_chain


def enrich_documents_with_reference_metadata(
    primary_docs: List[Document],
    all_docs: List[Document],
    reference_chain: dict
) -> List[Document]:
    """
    Enrich documents with metadata indicating if they are primary or referenced.

    Args:
        primary_docs: Documents from initial retrieval
        all_docs: All documents including referenced ones
        reference_chain: Dict mapping article refs to their referencing articles

    Returns:
        List of all documents with enriched metadata
    """
    primary_pages = set()
    for doc in primary_docs:
        page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
        primary_pages.add(page_key)

    enriched_docs = []
    for doc in all_docs:
        page_key = (doc.metadata.get('page'), doc.metadata.get('code'))

        if page_key in primary_pages:
            doc.metadata['source_type'] = 'primary'
            doc.metadata['referenced_by'] = []
        # Referenced docs already have metadata set in resolve_references_recursively

        enriched_docs.append(doc)

    return enriched_docs


class BalancedCodeRetriever(BaseRetriever):
    """
    Hybrid retriever that combines keyword search for article references
    with semantic search for general queries.

    Retriever híbrido que combina búsqueda por palabras clave para referencias
    de artículos con búsqueda semántica para consultas generales.
    """
    vectorstore: Chroma = Field(description="The Chroma vectorstore")
    k_per_code: int = Field(default=4, description="Number of documents to retrieve per code")
    code_filter: str = Field(default="both", description="Which codes to retrieve: 'both', 'nsr', or 'aci'")
    expand_references: bool = Field(default=False, description="Whether to expand article cross-references")
    max_expanded_articles: int = Field(default=15, description="Maximum articles after expansion")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve documents using hybrid approach:
        1. First, extract article references and do keyword search
        2. Also search for regulatory terms (LEY 400, ARTÍCULO, etc.)
        3. Then, do semantic search for remaining context
        4. Combine and deduplicate results
        """
        all_docs = []
        seen_pages = set()  # Track pages to avoid duplicates

        # Step 1a: Extract article references and do keyword search
        references = extract_article_references(query)
        if references:
            keyword_docs = keyword_search_documents(
                self.vectorstore,
                references,
                self.code_filter
            )
            for doc in keyword_docs:
                page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
                if page_key not in seen_pages:
                    seen_pages.add(page_key)
                    all_docs.append(doc)

        # Step 1b: Search for regulatory keywords in query
        # This helps find LEY 400 content, specific articles, or named sections
        import re
        query_lower = query.lower()
        regulatory_keywords = []

        # Check for regulatory document references
        if 'ley 400' in query_lower or 'ley400' in query_lower:
            regulatory_keywords.append('LEY 400')
        if 'métodos alternos' in query_lower or 'metodos alternos' in query_lower:
            regulatory_keywords.append('métodos alternos')
        if 'materiales alternos' in query_lower:
            regulatory_keywords.append('materiales alternos')

        # Check for "ARTÍCULO X" patterns (written out, not just numbers)
        articulo_match = re.search(r'art[íi]culo\s*(\d+)', query_lower)
        if articulo_match:
            regulatory_keywords.append(f'ARTÍCULO {articulo_match.group(1)}')

        # Check for cover/recubrimiento queries - add C.7.7 section search
        if 'recubrimiento' in query_lower or 'cover' in query_lower:
            regulatory_keywords.append('C.7.7')
            regulatory_keywords.append('CR7.7')
            regulatory_keywords.append('protección de concreto para el refuerzo')

        if regulatory_keywords:
            keyword_docs = keyword_search_documents(
                self.vectorstore,
                regulatory_keywords,
                self.code_filter
            )
            for doc in keyword_docs:
                page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
                if page_key not in seen_pages:
                    seen_pages.add(page_key)
                    all_docs.append(doc)

        # Step 2: Semantic search for additional context
        # Adjust k based on how many keyword results we found
        keyword_count = len(all_docs)
        remaining_k = max(2, self.k_per_code - keyword_count // 2)

        # PBDE mode: retrieve from ASCE-7, LATBSDC, and ACI-318
        if self.code_filter == "pbde":
            k = remaining_k  # Balanced across 3 codes
            for code_name in ["ASCE-7", "LATBSDC", "ACI-318"]:
                try:
                    code_docs = self.vectorstore.similarity_search(
                        query,
                        k=k,
                        filter={"code": code_name}
                    )
                    for doc in code_docs:
                        page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
                        if page_key not in seen_pages:
                            seen_pages.add(page_key)
                            all_docs.append(doc)
                except Exception:
                    pass
        else:
            # Colombian mode: NSR-10 and/or ACI-318
            retrieve_nsr = self.code_filter in ["both", "nsr"]
            retrieve_aci = self.code_filter in ["both", "aci"]

            # When filtering to single code, retrieve more documents
            k = remaining_k if self.code_filter == "both" else remaining_k * 2

            # Retrieve from NSR-10
            if retrieve_nsr:
                try:
                    nsr_docs = self.vectorstore.similarity_search(
                        query,
                        k=k,
                        filter={"code": "NSR-10"}
                    )
                    for doc in nsr_docs:
                        page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
                        if page_key not in seen_pages:
                            seen_pages.add(page_key)
                            all_docs.append(doc)
                except Exception:
                    pass

            # Retrieve from ACI-318
            if retrieve_aci:
                try:
                    aci_docs = self.vectorstore.similarity_search(
                        query,
                        k=k,
                        filter={"code": "ACI-318"}
                    )
                    for doc in aci_docs:
                        page_key = (doc.metadata.get('page'), doc.metadata.get('code'))
                        if page_key not in seen_pages:
                            seen_pages.add(page_key)
                            all_docs.append(doc)
                except Exception:
                    pass

        # Step 3: Expand references if enabled
        if self.expand_references and all_docs:
            primary_docs = list(all_docs)  # Keep copy of primary docs
            expanded_docs, reference_chain = resolve_references_recursively(
                self.vectorstore,
                all_docs,
                max_articles=self.max_expanded_articles
            )
            all_docs = enrich_documents_with_reference_metadata(
                primary_docs,
                expanded_docs,
                reference_chain
            )

        return all_docs


# =============================================================================
# CORE FUNCTIONS / FUNCIONES PRINCIPALES
# =============================================================================
def _check_and_pull_lfs():
    """Check if database is LFS pointer and attempt to pull."""
    import subprocess
    db_file = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    if not os.path.exists(db_file):
        return False, "Database file not found"

    file_size = os.path.getsize(db_file)
    # LFS pointer files are typically < 200 bytes
    if file_size < 1000:
        with open(db_file, 'r', errors='ignore') as f:
            content = f.read(100)
        if 'git-lfs' in content or 'version https://' in content:
            # Attempt to pull LFS files
            try:
                result = subprocess.run(
                    ['git', 'lfs', 'pull'],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                # Check if file size increased after pull
                new_size = os.path.getsize(db_file)
                if new_size > 1000:
                    return True, "LFS files pulled successfully"
                else:
                    return False, f"LFS pull did not resolve pointer. stderr: {result.stderr}"
            except FileNotFoundError:
                return False, "git-lfs not installed"
            except subprocess.TimeoutExpired:
                return False, "LFS pull timed out"
            except Exception as e:
                return False, f"LFS pull error: {str(e)}"
    return True, "Database OK"


@st.cache_resource
def load_vectordb():
    """Load the Chroma vector database from disk."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    # Check if database exists and is not a Git LFS pointer
    lfs_ok, lfs_message = _check_and_pull_lfs()
    if not lfs_ok:
        st.error(f"Database issue: {lfs_message}")
        st.info("The vector database may not have been downloaded correctly.")
        return None

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
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


def get_prompt_template(response_type: str, code_source: str, language: str, expand_references: bool = False) -> PromptTemplate:
    """
    Get the appropriate prompt template based on response type, code source, and language.

    Args:
        response_type: "dictionary" (citations only) or "recommendations" (with recommendations)
        code_source: "both", "nsr", or "aci"
        language: "es" or "en"
        expand_references: Whether expanded reference context is included
    """
    # Determine if dictionary mode (citations only) or recommendations mode
    is_dictionary = "citas" in response_type.lower() or "citations" in response_type.lower()
    if "Solo" in response_type or "only" in response_type.lower():
        is_dictionary = True
    if "Recomendaciones" in response_type or "Recommendations" in response_type:
        is_dictionary = False

    # Check for PBDE bundle (ASCE-7 + LATBSDC + ACI-318)
    is_pbde = "PBDE" in code_source or ("ASCE" in code_source and "LATBSDC" in code_source)

    if is_pbde:
        # PBDE prompts (English only)
        if is_dictionary:
            template = PROMPT_PBDE_DICTIONARY_EN
        else:
            template = PROMPT_PBDE_RECOMMENDATIONS_EN
    else:
        # Colombian codes (NSR-10 / ACI-318)
        is_nsr_only = "NSR" in code_source and "ACI" not in code_source
        is_aci_only = "ACI" in code_source and "NSR" not in code_source
        is_both = not is_nsr_only and not is_aci_only

        if is_dictionary:
            # Dictionary mode: citations only, no recommendations
            if is_both:
                template = PROMPT_DICTIONARY_BOTH_ES if language == "es" else PROMPT_DICTIONARY_BOTH_EN
            elif is_nsr_only:
                template = PROMPT_DICTIONARY_NSR_ES if language == "es" else PROMPT_DICTIONARY_NSR_EN
            else:  # is_aci_only
                template = PROMPT_DICTIONARY_ACI_ES if language == "es" else PROMPT_DICTIONARY_ACI_EN
        else:
            # Recommendations mode: with recommendations
            if is_both:
                template = PROMPT_NSR10_ACI318_ES if language == "es" else PROMPT_NSR10_ACI318_EN
            elif is_nsr_only:
                template = PROMPT_NSR10_SOLO_ES if language == "es" else PROMPT_NSR10_SOLO_EN
            else:  # is_aci_only
                template = PROMPT_ACI318_SOLO_ES if language == "es" else PROMPT_ACI318_SOLO_EN

    # Add expanded context instructions if reference expansion is enabled
    if expand_references:
        if language == "es":
            expanded_instructions = """

CONTEXTO EXPANDIDO CON REFERENCIAS CRUZADAS:
Los fragmentos del contexto están marcados como [PRIMARY] o [REFERENCED by X.X.X]:
- [PRIMARY]: Documentos encontrados directamente para tu consulta
- [REFERENCED by X.X.X]: Documentos referenciados por otros artículos (la cadena de referencias)

REGLAS PARA CONTEXTO EXPANDIDO:
1. Prioriza siempre las fuentes [PRIMARY] para responder la pregunta principal.
2. Usa las fuentes [REFERENCED] para proporcionar contexto adicional o clarificar referencias cruzadas.
3. Al citar una fuente [REFERENCED], menciona que proviene de una referencia cruzada (ej: "según C.7.10.4, referenciado por C.21.6.4.2").
4. Si la respuesta depende principalmente de fuentes [REFERENCED], aclara que la información viene de artículos relacionados.

"""
        else:
            expanded_instructions = """

EXPANDED CONTEXT WITH CROSS-REFERENCES:
Context fragments are marked as [PRIMARY] or [REFERENCED by X.X.X]:
- [PRIMARY]: Documents found directly for your query
- [REFERENCED by X.X.X]: Documents referenced by other articles (the reference chain)

RULES FOR EXPANDED CONTEXT:
1. Always prioritize [PRIMARY] sources to answer the main question.
2. Use [REFERENCED] sources to provide additional context or clarify cross-references.
3. When citing a [REFERENCED] source, mention it comes from a cross-reference (e.g., "per C.7.10.4, referenced by C.21.6.4.2").
4. If the answer relies primarily on [REFERENCED] sources, clarify that the information comes from related articles.

"""
        # Insert the expanded instructions before the context section
        template = template.replace("Contexto normativo:", expanded_instructions + "Contexto normativo:")
        template = template.replace("Normative context:", expanded_instructions + "Normative context:")

    # Add formula formatting instructions to all prompts
    if language == "es":
        formula_instructions = """
FORMATO DE FÓRMULAS MATEMÁTICAS:
Las fórmulas del contexto usan notación LaTeX con delimitadores $ y $$. MANTÉN estos delimitadores exactamente:
- Fórmulas en línea: $V_n = V_c + V_s$
- Fórmulas destacadas: $$V_n = V_c + V_s$$

REGLAS IMPORTANTES:
1. Copia las fórmulas del contexto TAL CUAL, incluyendo los símbolos $ o $$
2. NO mezcles delimitadores (NO escribas $V_n$ = ... sin cerrar correctamente)
3. Para fórmulas largas o importantes, usa $$ al inicio y $$ al final en líneas separadas
4. Asegúrate de que cada $ o $$ de apertura tenga su correspondiente cierre
5. Cuando ESCRIBAS fórmulas propias (no copiadas del contexto), SIEMPRE usa delimitadores $:
   - Variables con subíndices: $V_n$, $V_c$, $V_s$, $M_n$, $M_u$, $A_s$, $f'_c$, $f_y$
   - Ecuaciones: $V_n = V_c + V_s$, $φV_n ≥ V_u$
   - Factores: $φ$ (phi), $ρ$ (cuantía), $λ$ (factor de peso ligero)
6. NUNCA escribas variables como Vn, Vc, f'c sin delimitadores - SIEMPRE usa $V_n$, $V_c$, $f'_c$

"""
    else:
        formula_instructions = """
MATHEMATICAL FORMULA FORMAT:
Formulas in the context use LaTeX notation with $ and $$ delimiters. KEEP these delimiters exactly:
- Inline formulas: $V_n = V_c + V_s$
- Display formulas: $$V_n = V_c + V_s$$

IMPORTANT RULES:
1. Copy formulas from the context AS IS, including the $ or $$ symbols
2. DO NOT mix delimiters (DO NOT write $V_n$ = ... without proper closing)
3. For long or important formulas, use $$ at the start and $$ at the end on separate lines
4. Ensure every opening $ or $$ has its corresponding closing delimiter
5. When WRITING your own formulas (not copied from context), ALWAYS use $ delimiters:
   - Variables with subscripts: $V_n$, $V_c$, $V_s$, $M_n$, $M_u$, $A_s$, $f'_c$, $f_y$
   - Equations: $V_n = V_c + V_s$, $φV_n ≥ V_u$
   - Factors: $φ$ (phi), $ρ$ (reinforcement ratio), $λ$ (lightweight factor)
6. NEVER write variables like Vn, Vc, f'c without delimiters - ALWAYS use $V_n$, $V_c$, $f'_c$

"""
    # Insert formula instructions near the beginning
    template = template.replace("REGLAS ESTRICTAS:", formula_instructions + "REGLAS ESTRICTAS:")
    template = template.replace("STRICT RULES:", formula_instructions + "STRICT RULES:")
    template = template.replace("INSTRUCCIONES:", formula_instructions + "INSTRUCCIONES:")
    template = template.replace("INSTRUCTIONS:", formula_instructions + "INSTRUCTIONS:")

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


def query_with_sources(vectordb, llm, question: str, response_type: str, code_source: str, language: str, chat_history: list, expand_references: bool = False) -> dict:
    """
    Query the vectorstore and generate a response with sources.

    Args:
        vectordb: Chroma vector database
        llm: Language model
        question: User's question
        response_type: "dictionary" (citations only) or "recommendations" (with recommendations)
        code_source: Code source selection from UI (e.g., "NSR-10 + ACI-318", "Solo NSR-10", "Solo ACI-318")
        language: "es" or "en"
        chat_history: List of previous messages for conversational context
        expand_references: Whether to expand cross-references in retrieved documents

    Returns:
        dict with "answer" and "source_documents"
    """
    # Build contextualized query for better retrieval on follow-up questions
    search_query = build_contextualized_query(question, chat_history, llm)

    # Determine code filter based on code_source selection
    if "PBDE" in code_source or ("ASCE" in code_source and "LATBSDC" in code_source):
        code_filter = "pbde"
    elif "ACI" in code_source and "NSR" not in code_source:
        code_filter = "aci"
    elif "NSR" in code_source and "ACI" not in code_source:
        code_filter = "nsr"
    else:
        code_filter = "both"

    # Create retriever with appropriate code filter and expansion setting
    retriever = BalancedCodeRetriever(
        vectorstore=vectordb,
        k_per_code=4,  # 4 from each code when both, 8 when single code
        code_filter=code_filter,
        expand_references=expand_references,
        max_expanded_articles=15
    )

    # Retrieve relevant documents using contextualized query
    source_documents = retriever.invoke(search_query)

    # Build context from documents with metadata (including source type if expanded)
    context_parts = []
    for doc in source_documents:
        code = doc.metadata.get('code', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        source_type = doc.metadata.get('source_type', 'primary')
        referenced_by = doc.metadata.get('referenced_by', [])

        # Convert LaTeX delimiters to Streamlit-compatible format
        # This ensures LLM sees $...$ and $$...$$ which it will preserve in output
        clean_content = convert_latex_delimiters(doc.page_content)

        # Add source type annotation if expansion is enabled
        if expand_references:
            if source_type == 'referenced' and referenced_by:
                ref_info = f"[REFERENCED by {', '.join(referenced_by)}]"
            else:
                ref_info = "[PRIMARY]"
            context_parts.append(f"[code={code}, page={page}] {ref_info}\n{clean_content}")
        else:
            context_parts.append(f"[code={code}, page={page}]\n{clean_content}")

    context = "\n\n---\n\n".join(context_parts)

    # Format chat history
    formatted_history = format_chat_history(chat_history, language)

    # Get prompt and generate response (with expanded context instructions if enabled)
    prompt = get_prompt_template(response_type, code_source, language, expand_references)
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

        # Response type selector (Dictionary vs Recommendations)
        response_type = st.radio(
            txt['response_type_label'],
            options=txt['response_type_options'],
            index=1,  # Default to "Citations + Recommendations"
            help=txt['response_type_help']
        )

        st.divider()

        # Code source selector (NSR-10, ACI-318, Both, or PBDE)
        # PBDE option only available in English
        if lang == "en":
            code_source_options = txt['code_source_options']  # Includes PBDE
        else:
            # Spanish: filter out PBDE option
            code_source_options = [opt for opt in txt['code_source_options'] if "PBDE" not in opt]

        code_source = st.radio(
            txt['code_source_label'],
            options=code_source_options,
            index=0,  # Default to "NSR-10 + ACI-318"
            help=txt['code_source_help']
        )

        st.divider()

        # Advanced options section
        st.markdown(f"#### {txt['advanced_options_header']}")
        expand_references = st.toggle(
            txt['expand_references_label'],
            value=False,
            help=txt['expand_references_help']
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
                            st.markdown(normalize_latex_output(msg["content"]))

                            # Show sources grouped by code (compact format)
                            # Always show expander to display sources or indicate if none found
                            with st.expander(txt['sources_expander']):
                                sources_list = msg.get("sources", [])
                                if not sources_list:
                                    st.info("No se encontraron fuentes / No sources found")
                                else:
                                    # Show reference chain tree if cross-references were expanded
                                    chain_tree = build_reference_chain_tree(sources_list, txt)
                                    if chain_tree:
                                        st.markdown(f"**🔗 {txt['reference_chain_header']}**")
                                        st.code(chain_tree, language=None)
                                        st.divider()

                                    # Compact list of sources - separate by code
                                    nsr_sources = [s for s in sources_list if s.get("code") == "NSR-10"]
                                    aci_sources = [s for s in sources_list if s.get("code") == "ACI-318"]
                                    asce_sources = [s for s in sources_list if s.get("code") == "ASCE-7"]
                                    latbsdc_sources = [s for s in sources_list if s.get("code") == "LATBSDC"]
                                    known_codes = {"NSR-10", "ACI-318", "ASCE-7", "LATBSDC"}
                                    other_sources = [s for s in sources_list if s.get("code") not in known_codes]

                                    # Helper function to display sources in compact format
                                    def display_sources_compact(src_list, header):
                                        if not src_list:
                                            return
                                        st.markdown(f"**{header}**")

                                        # Separate primary and referenced
                                        primary = [s for s in src_list if s.get("source_type", "primary") == "primary"]
                                        referenced = [s for s in src_list if s.get("source_type") == "referenced"]

                                        # Display primary sources
                                        for source in primary:
                                            sections = source.get("sections", [])
                                            section_str = f" — §{', §'.join(sections)}" if sections else ""
                                            st.markdown(f"- {txt['page_label']} {source['page']}{section_str}")

                                        # Display referenced sources with indicator
                                        for source in referenced:
                                            sections = source.get("sections", [])
                                            section_str = f" — §{', §'.join(sections)}" if sections else ""
                                            ref_by = source.get("referenced_by", [])
                                            ref_str = f" *(↩ {', '.join(ref_by)})*" if ref_by else " *(ref)*"
                                            st.markdown(f"- {txt['page_label']} {source['page']}{section_str}{ref_str}")

                                    # Track if we've displayed any sources (for spacing)
                                    displayed_any = False

                                    # ASCE-7 sources (PBDE mode)
                                    if asce_sources:
                                        display_sources_compact(asce_sources, txt.get('asce_sources_header', '📗 ASCE-7 Sources'))
                                        displayed_any = True

                                    # LATBSDC sources (PBDE mode)
                                    if latbsdc_sources:
                                        if displayed_any:
                                            st.write("")
                                        display_sources_compact(latbsdc_sources, txt.get('latbsdc_sources_header', '📕 LATBSDC Sources'))
                                        displayed_any = True

                                    # NSR-10 sources
                                    if nsr_sources:
                                        if displayed_any:
                                            st.write("")
                                        display_sources_compact(nsr_sources, txt['nsr_sources_header'])
                                        displayed_any = True

                                    # ACI-318 sources
                                    if aci_sources:
                                        if displayed_any:
                                            st.write("")
                                        display_sources_compact(aci_sources, txt['aci_sources_header'])
                                        displayed_any = True

                                    # Other sources
                                    if other_sources:
                                        if displayed_any:
                                            st.write("")
                                        st.markdown(f"**{txt['other_sources_header']}**")
                                        for source in other_sources:
                                            sections = source.get("sections", [])
                                            section_str = f" — §{', §'.join(sections)}" if sections else ""
                                            st.markdown(f"- {txt['page_label']} {source['page']}{section_str}")

                            # Feedback rating buttons
                            interaction_id = msg.get("interaction_id")
                            if interaction_id and not msg.get("rated", False):
                                # Check if user clicked thumbs down and wants to add comment
                                pending_feedback_key = f"pending_feedback_{msg_idx}"

                                if st.session_state.get(pending_feedback_key):
                                    # Show comment form for negative feedback
                                    st.caption(txt['feedback_comment_prompt'])
                                    comment = st.text_area(
                                        txt['feedback_placeholder'],
                                        key=f"comment_{msg_idx}",
                                        height=80
                                    )
                                    col_submit, col_skip = st.columns([1, 1])
                                    with col_submit:
                                        if st.button(txt['feedback_submit'], key=f"submit_feedback_{msg_idx}", type="primary"):
                                            session_id = get_session_id()
                                            if update_rating(session_id, interaction_id, 1, comment):
                                                st.session_state.messages[msg_idx]["rated"] = True
                                                st.session_state.messages[msg_idx]["rating"] = 1
                                                st.session_state[pending_feedback_key] = False
                                                st.rerun()
                                    with col_skip:
                                        if st.button(txt['feedback_skip'], key=f"skip_feedback_{msg_idx}"):
                                            session_id = get_session_id()
                                            if update_rating(session_id, interaction_id, 1, ""):
                                                st.session_state.messages[msg_idx]["rated"] = True
                                                st.session_state.messages[msg_idx]["rating"] = 1
                                                st.session_state[pending_feedback_key] = False
                                                st.rerun()
                                else:
                                    # Show rating buttons
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
                                            # Show comment form instead of immediate submission
                                            st.session_state[pending_feedback_key] = True
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

        # Determine spinner text based on code source
        if "PBDE" in code_source:
            spinner_text = txt.get('pbde_spinner_text', 'Querying ASCE-7, LATBSDC, and ACI-318 codes...')
        else:
            spinner_text = txt['spinner_text']

        with st.spinner(spinner_text):
            # Pass previous messages for conversational context
            result = query_with_sources(vectordb, llm, query_to_process, response_type, code_source, lang, previous_messages, expand_references)

            answer = result["answer"]
            sources = []

            # Extract source information with code and reference metadata
            for doc in result.get("source_documents", []):
                page = doc.metadata.get("page", "N/A")
                code = doc.metadata.get("code", "Unknown")
                source_type = doc.metadata.get("source_type", "primary")
                referenced_by = doc.metadata.get("referenced_by", [])
                sections = extract_section_headers(doc.page_content)
                sources.append({
                    "page": page,
                    "code": code,
                    "sections": sections,
                    "source_type": source_type,
                    "referenced_by": referenced_by
                })

            # Separate sources by code for logging
            nsr_sources = [s for s in sources if s.get("code") == "NSR-10"]
            aci_sources = [s for s in sources if s.get("code") == "ACI-318"]

            # Compose mode string for logging
            mode_for_log = f"{response_type} | {code_source}"

            # Log interaction to Google Sheets
            session_id = get_session_id()
            interaction_id = log_interaction(
                session_id=session_id,
                query=query_to_process,
                response=answer,
                sources_nsr=nsr_sources,
                sources_aci=aci_sources,
                mode=mode_for_log,
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
