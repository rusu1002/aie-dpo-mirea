# import sys
# from pathlib import Path

# ROOT = Path(__file__).resolve().parents[2]

# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# import requests
# import streamlit as st


# API_URL = "http://localhost:8000"


# # ==================================================
# # PAGE CONFIG
# # ==================================================

# st.set_page_config(
#     page_title="Финансовый RAG Ассистент",
#     page_icon="💰",
#     layout="wide"
# )


# # ==================================================
# # SIDEBAR
# # ==================================================

# with st.sidebar:

#     st.title("Статус Системы")

#     try:

#         health = requests.get(f"{API_URL}/health", timeout=5).json()

#         st.success("API Online")

#         st.subheader("LLM")
#         st.json(health["llm"])

#         st.subheader("Reranker")
#         st.json(health["reranker"])

#         st.caption(
#             f"Uptime: {health['uptime_seconds']} sec"
#         )

#     except Exception:

#         st.error(
#             "FastAPI service is unavailable"
#         )

#     st.divider()

#     st.markdown(
#         """
#         ### Описание

#         Финансовая вопросно-ответная интеллектуальная система

#         Архитектура:

#         - Dense Retrieval
#         - BM25 Retrieval
#         - Hybrid Search
#         - Reranking
#         - LLM Generation
#         """
#     )


# # ==================================================
# # HEADER
# # ==================================================

# st.title("💰 Финансовый RAG Ассистент")

# st.markdown(
#     """
#     Задайте вопрос на финансовую тему и получите 
#     ответ, сгенерированный интеллектуальной системой.
#     """
# )

# st.divider()


# # ==================================================
# # CHAT HISTORY
# # ==================================================

# if "messages" not in st.session_state:
#     st.session_state.messages = []


# for msg in st.session_state.messages:

#     with st.chat_message(msg["role"]):

#         st.markdown(msg["content"])

#         if (
#             msg["role"] == "assistant"
#             and msg.get("sources")
#         ):

#             with st.expander("Sources"):

#                 for source in msg["sources"]:
#                     st.write(source)


# # ==================================================
# # INPUT
# # ==================================================

# question = st.chat_input(
#     "Задайте интересующий вопрос..."
# )

# if question:

#     st.session_state.messages.append(
#         {
#             "role": "user",
#             "content": question
#         }
#     )

#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.chat_message("assistant"):

#         try:

#             with st.spinner(
#                 "Генерация ответа..."
#             ):

#                 response = requests.post(
#                     f"{API_URL}/predict",
#                     json={
#                         "question": question
#                     },
#                     timeout=300
#                 )

#                 response.raise_for_status()

#                 result = response.json()

#             answer = result["answer"]

#             sources = result.get(
#                 "sources",
#                 []
#             )

#             latency_ms = result.get(
#                 "latency_ms",
#                 0
#             )

#             st.markdown(answer)

#             if sources:

#                 with st.expander("Sources"):

#                     for source in sources:
#                         st.write(source)

#             st.caption(
#                 f"Задержка: {latency_ms:.0f} ms"
#             )

#             st.session_state.messages.append(
#                 {
#                     "role": "assistant",
#                     "content": answer,
#                     "sources": sources
#                 }
#             )

#         except Exception as e:

#             error_message = (
#                 f"❌ Request failed: {e}"
#             )

#             st.error(error_message)

#             st.session_state.messages.append(
#                 {
#                     "role": "assistant",
#                     "content": error_message
#                 }
#             )

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import streamlit as st
from datetime import datetime


API_URL = "http://localhost:8000"


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Финансовый RAG Ассистент",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
    /* Основные стили */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Заголовок */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Чат-сообщения */
    .user-message {
        background: linear-gradient(135deg, #a8b5f0 0%, #c9a8d4 100%);
        color: #2c3e50;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .assistant-message {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e8e8e8;
    }
    
    /* Sidebar */
    .sidebar-section {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .sidebar-section h3 {
        color: #667eea;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
        padding-bottom: 0.5rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online {
        background: #4CAF50;
        box-shadow: 0 0 10px rgba(76,175,80,0.5);
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background: #f44336;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(76,175,80,0.4); }
        70% { box-shadow: 0 0 10px 0 rgba(76,175,80,0); }
        100% { box-shadow: 0 0 0 0 rgba(76,175,80,0); }
    }
    
    /* Метрики */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e8e8e8;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        font-weight: 500;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
    }
    
    /* Кнопки */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 500;
        color: #667eea;
    }
    
    /* ============================================
       CHAT INPUT - МАКСИМАЛЬНО АГРЕССИВНЫЙ ПОДХОД
       ============================================ */
    
    /* Переопределяем всё для поля ввода */
    .stChatInput input,
    .stChatInput div input,
    div.stChatInput input,
    [data-testid="stChatInput"] input,
    div[data-testid="stChatInput"] input,
    .stChatInput > div > div > input,
    .stChatInput div[data-baseweb] input {
        border: 2px solid #667eea !important;
        border-radius: 25px !important;
        outline: none !important;
        box-shadow: none !important;
        background: white !important;
    }
    
    /* Фокус - ТОЛЬКО сине-фиолетовый, БЕЗ красного */
    .stChatInput input:focus,
    .stChatInput div input:focus,
    div.stChatInput input:focus,
    [data-testid="stChatInput"] input:focus,
    div[data-testid="stChatInput"] input:focus,
    .stChatInput > div > div > input:focus,
    .stChatInput div[data-baseweb] input:focus {
        border: 2px solid #764ba2 !important;
        outline: 3px solid #667eea !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 5px rgba(102, 126, 234, 0.15) !important;
        background: white !important;
    }
    
    /* Убираем все возможные outline от Streamlit */
    .stChatInput:focus,
    .stChatInput:focus-within,
    [data-testid="stChatInput"]:focus,
    [data-testid="stChatInput"]:focus-within {
        outline: none !important;
        box-shadow: none !important;
        border: none !important;
    }
    
    /* Убираем красный outline от браузера */
    .stChatInput input::-moz-focus-inner,
    .stChatInput input::-moz-focus-outer {
        border: none !important;
        outline: none !important;
    }
    
    /* ============================================
       КНОПКА ОТПРАВКИ
       ============================================ */
    
    .stChatInput button,
    .stChatInput div button,
    div.stChatInput button,
    [data-testid="stChatInput"] button,
    div[data-testid="stChatInput"] button,
    button[data-testid="stChatInputSendButton"],
    .stChatInput > div > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 50% !important;
        width: 48px !important;
        height: 48px !important;
        color: white !important;
        outline: none !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatInput button:hover,
    [data-testid="stChatInput"] button:hover,
    button[data-testid="stChatInputSendButton"]:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.5) !important;
        outline: none !important;
    }
    
    .stChatInput button:focus,
    [data-testid="stChatInput"] button:focus,
    button[data-testid="stChatInputSendButton"]:focus {
        outline: 3px solid #667eea !important;
        outline-offset: 2px !important;
        box-shadow: 0 0 0 5px rgba(102, 126, 234, 0.2) !important;
    }
    
    .stChatInput button svg,
    [data-testid="stChatInput"] button svg,
    button[data-testid="stChatInputSendButton"] svg {
        color: white !important;
        fill: white !important;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #667eea, transparent);
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    
    .badge-primary {
        background: #e8eaf6;
        color: #667eea;
    }
    
    .badge-success {
        background: #e8f5e9;
        color: #4CAF50;
    }
    /* Экстренное переопределение через * (звездочка) */
    *:focus {
        outline-color: #667eea !important;
    }

    .stChatInput *:focus {
        outline-color: #667eea !important;
        border-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    
    # Logo/Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">💰</div>
        <h2 style="color: #667eea; margin: 0;">RAG Ассистент</h2>
        <p style="color: #666; font-size: 0.9rem;">Финансовый интеллект</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # System Status
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📊 Статус Системы")
    
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">API</div>
                <div>
                    <span class="status-indicator status-online"></span>
                    <span class="metric-value" style="color: #4CAF50;">Online</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            uptime_min = health['uptime_seconds'] // 60
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Uptime</div>
                <div class="metric-value">{uptime_min} мин</div>
            </div>
            """, unsafe_allow_html=True)
        
        # LLM Status
        st.markdown("#### 🤖 Модели")
        col1, col2 = st.columns(2)
        with col1:
            llm_status = health['llm'].get('llm_available', False)
            llm_models = health['llm'].get('llms', ['N/A'])
            llm_model_name = llm_models[0] if llm_models else 'N/A'
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.8rem; color: #666;">LLM</div>
                <div style="font-weight: 600; color: #333; font-size: 0.75rem; word-break: break-all;">{llm_model_name}</div>
                <div style="font-size: 0.75rem; color: {'#4CAF50' if llm_status else '#f44336'};">
                    {'✅ Active' if llm_status else '❌ Inactive'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            reranker_status = health['reranker'].get('reranker_available', False)
            reranker_models = health['reranker'].get('rerankers', [])
            reranker_model_name = reranker_models[0] if reranker_models else 'N/A'
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 8px; text-align: center;">
                <div style="font-size: 0.8rem; color: #666;">Reranker</div>
                <div style="font-weight: 600; color: #333; font-size: 0.75rem; word-break: break-all;">{reranker_model_name}</div>
                <div style="font-size: 0.75rem; color: {'#4CAF50' if reranker_status else '#f44336'};">
                    {'✅ Active' if reranker_status else '❌ Inactive'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception:
        st.markdown("""
        <div style="background: #ffebee; padding: 1rem; border-radius: 10px; border-left: 4px solid #f44336;">
            <div style="color: #c62828; font-weight: 500;">❌ API Offline</div>
            <div style="color: #666; font-size: 0.9rem;">Сервис недоступен</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Architecture Info
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 🏗️ Архитектура")
    
    arch_items = [
        ("Dense Retrieval", "🔍"),
        ("BM25 Retrieval", "📚"),
        ("Hybrid Search", "🔄"),
        ("Reranking", "🎯"),
        ("LLM Generation", "🧠")
    ]
    
    for item, icon in arch_items:
        st.markdown(f"""
        <div style="display: flex; align-items: center; padding: 0.3rem 0;">
            <span style="margin-right: 0.5rem;">{icon}</span>
            <span style="color: #444; font-size: 0.9rem;">{item}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Actions
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚡ Быстрые вопросы")
    
    quick_questions = [
        "Как мошенники крадут реквизиты при оплате онлайн?",
        "Что делать при СМС о блокировке банковской карты?",
        "Что выгоднее: вклад или инвестиции в ценные бумаги?",
        "Что такое процентная ставка ЦБ?"
    ]
    
    for q in quick_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.quick_question = q
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="main-header">
    <h1>💰 Финансовый RAG Ассистент</h1>
    <p>Интеллектуальная система для ответов на финансовые вопросы</p>
    <div style="margin-top: 0.5rem;">
        <span class="badge badge-primary">RAG</span>
        <span class="badge badge-primary">Hybrid Search</span>
        <span class="badge badge-success">Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quick_question" in st.session_state:
    question = st.session_state.quick_question
    del st.session_state.quick_question
else:
    question = None

# Display chat messages
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            <strong>👤 Вы</strong><br>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Ассистент</strong><br>
            {msg["content"]}
        </div>
        """, unsafe_allow_html=True)
        
        if msg.get("sources"):
            with st.expander(f"📚 Источники ({len(msg['sources'])})"):
                for source in msg["sources"]:
                    st.markdown(f"• {source}")

# ==================================================
# INPUT
# ==================================================

if question is None:
    question = st.chat_input("💬 Задайте интересующий вопрос...")

if question:
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    
    # Display user message
    st.markdown(f"""
    <div class="user-message">
        <strong>👤 Вы</strong><br>
        {question}
    </div>
    """, unsafe_allow_html=True)
    
    # Get assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("🧠 Генерация ответа..."):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"question": question},
                    timeout=300
                )
                response.raise_for_status()
                result = response.json()
            
            answer = result["answer"]
            sources = result.get("sources", [])
            latency_ms = result.get("latency_ms", 0)
            
            # Display assistant message
            st.markdown(f"""
            <div class="assistant-message">
                <strong>🤖 Ассистент</strong><br>
                {answer}
            </div>
            """, unsafe_allow_html=True)
            
            # Sources
            if sources:
                with st.expander(f"📚 Источники ({len(sources)})"):
                    for source in sources:
                        st.markdown(f"• {source}")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("⏱️ Задержка", f"{latency_ms:.0f} ms")
            with col2:
                st.metric("📊 Источники", len(sources))
            with col3:
                st.metric("📝 Длина ответа", len(answer.split()))
            
            # Save to session
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
        except Exception as e:
            error_message = f"❌ Ошибка: {str(e)}"
            st.error(error_message)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })
    
    st.rerun()