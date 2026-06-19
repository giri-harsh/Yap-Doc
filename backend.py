
import os
import tempfile
from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import (
        PyPDFLoader,
        Docx2txtLoader,
        TextLoader,
    )
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Model name maps exposed via the API
GROQ_MODELS = {
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "mixtral-8x7b": "mixtral-8x7b-32768",
}
GEMINI_MODELS = {
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
}

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


# ──────────────────────────────────────────────────────────────────────────────
# Document loading & chunking
# ──────────────────────────────────────────────────────────────────────────────

def load_document(filename: str, file_bytes: bytes):
    
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Free, local embeddings — no extra API key needed
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(chunks)


# ──────────────────────────────────────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_context(query: str, vectorstore, k: int = 5) -> str:
    """Return top-k chunks joined as a single context string."""
    if vectorstore is None:
        return ""
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n---\n\n".join(d.page_content for d in docs)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Yap-Doc, a helpful AI assistant that answers questions about documents.
When context from a document is provided, ground your answer in that context.
If no context is available, answer from general knowledge.
Be concise, accurate, and helpful. Format your response in clear Markdown."""

DEEP_RESEARCH_SUFFIX = """
Additionally, since Deep Research mode is enabled:
- Provide a thorough, multi-paragraph analysis.
- List key insights, caveats, and follow-up questions the user might consider.
- Cite specific sections from the provided context where relevant."""


def build_messages(
    query: str,
    context: str,
    chat_history: list,
    deep_research: bool,
) -> tuple:
    system = SYSTEM_PROMPT
    if deep_research:
        system += DEEP_RESEARCH_SUFFIX

    messages = []

    # Include last 6 turns of history for context window efficiency
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_content = query
    if context:
        user_content = f"**Document context:**\n{context}\n\n**Question:** {query}"

    messages.append({"role": "user", "content": user_content})
    return system, messages


# ──────────────────────────────────────────────────────────────────────────────
# Model calls
# ──────────────────────────────────────────────────────────────────────────────

def call_groq(system: str, messages: list, deep_research: bool) -> str:
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "⚠️ `GROQ_API_KEY` not found in environment. Add it to your `.env` file (or Render's env vars)."

    client = Groq(api_key=api_key)

    full_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,
        messages=full_messages,
        max_tokens=2048 if deep_research else 1024,
        temperature=0.3,
    )
    return response.choices[0].message.content


def call_gemini(system: str, messages: list, deep_research: bool) -> str:
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "⚠️ `GOOGLE_API_KEY` not found in environment. Add it to your `.env` file (or Render's env vars)."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=DEFAULT_GEMINI_MODEL,
        system_instruction=system,
    )

    # Convert to Gemini format (roles: user / model)
    history = []
    for msg in messages[:-1]:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=history)
    last_user = messages[-1]["content"]

    generation_config = genai.types.GenerationConfig(
        max_output_tokens=2048 if deep_research else 1024,
        temperature=0.3,
    )
    response = chat.send_message(last_user, generation_config=generation_config)
    return response.text


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_answer(
    query: str,
    model_choice: str,          # "Groq" or "Google Gemini"
    vectorstore,                # FAISS instance or None
    deep_research: bool = False,
    chat_history: Optional[list] = None,
) -> str:
    if chat_history is None:
        chat_history = []

    k = 8 if deep_research else 5
    context = retrieve_context(query, vectorstore, k=k)
    system, messages = build_messages(query, context, chat_history, deep_research)

    try:
        if model_choice == "Groq":
            return call_groq(system, messages, deep_research)
        else:
            return call_gemini(system, messages, deep_research)
    except Exception as e:
        return f"❌ Error from {model_choice}: `{e}`"
