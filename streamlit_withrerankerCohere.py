import os
import sys
import httpx
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv, find_dotenv
import streamlit as st
from supabase.client import create_client, Client
import json 
from datetime import datetime

# 🚨 COHERE ADDED: Import Cohere Client
try:
    import cohere
except ImportError:
    st.error("❌ خطای نصب: لطفاً 'pip install cohere' را اجرا کنید.")
    sys.exit(1)

# ---------------- Streamlit Configuration (MUST BE FIRST COMMAND) ----------------
st.set_page_config(page_title="Cohere RAG Chatbot (Multilingual/1024 Dim)", layout="wide")

# ---------------- ENV & CONFIG ----------------
load_dotenv(find_dotenv(), override=True)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
COHERE_API_KEY = os.environ.get("COHERE_API_KEY") 

# 🚨 مدل‌های Cohere Multilingual برای فارسی
EMBED_MODEL = "embed-multilingual-v3.0"  
EMBED_DIM = 1024                         

# 🚨🚨 مدل ریرنکر چندزبانه برای دقت بالاتر
RERANK_MODEL = "rerank-multilingual-v3.0" 
GEMINI_MODEL = "gemini-2.5-flash" 

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# 🚨 تنظیمات Reranking
TOP_K_RETRIEVAL = 15 
# 🚨🚨 تنظیم نهایی: 5 چانک مرتبط‌ترین (بعد از Reranking) به Gemini ارسال می‌شود
TOP_K_RERANK = 5      

if not all([GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, COHERE_API_KEY]):
    st.error("❌ تنظیمات ناقص: کلیدهای API و Supabase را در فایل .env تنظیم کنید.")
    
try:
    # Supabase Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    st.error(f"❌ خطای اتصال به Supabase: {e}")
    sys.exit(1)

# 🚨 Cohere Client
@st.cache_resource
def load_cohere_client():
    try:
        co = cohere.Client(api_key=COHERE_API_KEY)
        st.success("✅ Cohere Client Initialized.", icon="🧠")
        return co
    except Exception as e:
        st.error(f"❌ خطای اتصال به Cohere. RAG غیرفعال شد: {e}")
        return None

COHERE_CLIENT = load_cohere_client()
RERANKER_ACTIVE = (COHERE_CLIENT is not None)

# ----------------------------------------------------------------------

# ---------------- COHERE EMBEDDING CLIENT ----------------

class CohereEmbedClient:
    """کلاینت Embedding برای Cohere API."""
    
    def __init__(self, cohere_client: cohere.Client):
        self.co = cohere_client
        self.model = EMBED_MODEL
        self.dim = EMBED_DIM 

    def embed(self, text: str) -> List[float]:
        if not self.co:
            return [0.0] * self.dim
            
        try:
            response = self.co.embed(
                texts=[text],
                model=self.model,
                input_type="search_query"
            )
            vector = response.embeddings[0]
            
            if len(vector) != self.dim:
                st.warning(f"⚠️ Cohere API returned {len(vector)} dims instead of {self.dim}. Using received dimension.")
                self.dim = len(vector)
                
            return vector
            
        except Exception as e:
            st.error(f"❌ Cohere Embed API Error: {e}")
            return [0.0] * self.dim

# ---------------- LLM Client (Gemini Generation) ----------------
class GeminiClient:
    """کلاینت Generation برای Gemini API."""
    
    GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = GEMINI_MODEL 
        self.api_path = f"{self.GEMINI_API_BASE_URL}/v1beta/models/{self.model}:generateContent"
        self.full_url = self.api_path 
        self.headers = {"Content-Type": "application/json"}
        self.client = httpx.Client(timeout=180)

    def generate(self, system_prompt: str, history: List[Dict[str, str]], user_prompt: str) -> str:
        
        api_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            if msg.get("content"):
                api_history.append({"role": role, "parts": [{"text": msg["content"]}]})

        contents = [
            {"role": "user", "parts": [{"text": system_prompt}]},
            {"role": "model", "parts": [{"text": "باشه، درک شد."}]}
        ]
        
        contents.extend(api_history[1:]) 
        contents.append({"role": "user", "parts": [{"text": user_prompt}]})
        
        payload = {"contents": contents}
        params = {"key": self.api_key} 

        try:
            r = self.client.post(self.full_url, headers=self.headers, json=payload, params=params)
            
            if not r.is_success:
                msg = r.json().get("error", {}).get("message", r.text)
                st.error(f"Gemini API Error {r.status_code}: {msg}")
                return "خطا در برقراری ارتباط با مدل Gemini. (لطفاً GOOGLE_API_KEY را چک کنید)"
                
            data = r.json()
            return data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or "No response."
        except Exception as e:
            st.error(f"LLM Connection Error: {e}")
            return "خطای شبکه یا اتصال به Gemini."

# ---------------- COHERE RERANKING FUNCTION ----------------

def rerank_documents(query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """اعمال بازرتبه‌بندی با استفاده از Cohere Rerank API و مرتب‌سازی نهایی بر اساس نمره ریرنک."""
    
    if not RERANKER_ACTIVE or not retrieved_chunks:
        # در صورت غیرفعال بودن، بر اساس Similarity اولیه مرتب می‌کنیم
        retrieved_chunks.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        return retrieved_chunks[:top_k]

    try:
        documents_text = [chunk['content'] for chunk in retrieved_chunks]
        
        response = COHERE_CLIENT.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=documents_text,
            top_n=top_k
        )
        
        final_ranked_chunks = []
        chunk_map = {i: chunk for i, chunk in enumerate(retrieved_chunks)}
        
        for rank_result in response.results:
            original_index = rank_result.index
            chunk = chunk_map[original_index]
            chunk['rerank_score'] = rank_result.relevance_score
            final_ranked_chunks.append(chunk)

        # 🚨 تضمین مرتب‌سازی: مرتبط‌ترین چانک (بالاترین نمره) در چانک 1 قرار می‌گیرد.
        final_ranked_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        st.info(f"✅ Cohere Reranking completed using **{RERANK_MODEL}**. Selecting top {top_k}.", icon="✨")
        
        return final_ranked_chunks
        
    except Exception as e:
        st.error(f"❌ Cohere Rerank API Error. Falling back to Similarity: {e}")
        retrieved_chunks.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
        return retrieved_chunks[:top_k]


# ---------------- RAG LOGIC ----------------

@st.cache_resource
def get_clients():
    """ایجاد یکبارۀ کلاینت‌ها و کش کردن آنها."""
    # بررسی کنید که آیا COHERE_CLIENT با موفقیت بارگذاری شده است یا خیر
    if COHERE_CLIENT is None:
        st.error("❌ Cohere Client ناموفق بود. RAG غیرفعال است.")
        # ایجاد کلاینت‌های جایگزین برای جلوگیری از خطا
        return CohereEmbedClient(None), GeminiClient(GOOGLE_API_KEY)
        
    return CohereEmbedClient(COHERE_CLIENT), GeminiClient(GOOGLE_API_KEY)

cohere_embed_service, gemini_service = get_clients()


def retrieve_rag_context(query: str) -> str:
    """بازیابی زمینه RAG را به صورت سنکرون انجام می‌دهد و Reranking را اعمال می‌کند."""
    
    match_count = TOP_K_RETRIEVAL 
    
    try:
        # 1. تولید Embedding کوئری (1024 بُعد)
        qvec = cohere_embed_service.embed(query)
        
        current_dim = cohere_embed_service.dim 
        if not qvec or len(qvec) != current_dim or all(e == 0.0 for e in qvec):
            return f"RAG disabled: Could not generate a valid {current_dim}-dimension embedding vector using Cohere."

        # 2. جستجوی اولیه در Supabase
        res = supabase.rpc(
            "match_site_pages",
            {"query_embedding": qvec, "match_count": match_count}
        ).execute()

        if not res.data:
            return "No relevant documentation found."
        
        initial_chunks = res.data
        
        # 3. 🚨 اعمال RERANKING (5 چانک)
        final_ranked_chunks = rerank_documents(query, initial_chunks, TOP_K_RERANK) 
        
        # 4. ساخت Context نهایی از نتایج بازرتبه‌بندی شده
        chunks = []
        for i, row in enumerate(final_ranked_chunks):
            similarity = row.get("similarity", "N/A")
            rerank_score = row.get("rerank_score", "N/A")
            
            source_info = row.get("url", "local").split('//')[-1]
            title = row.get("title", "Untitled")
            content = row.get("content", "")
            
            # نمایش نمرات با دقت 3 رقم اعشار
            score_info = f"Sim: {similarity:.3f}, Rerank: {rerank_score:.3f}" if rerank_score != "N/A" else f"Sim: {similarity:.3f}"
            
            # چانک 1 مرتبط‌ترین چانک است.
            chunks.append(f"--- Chunk {i+1} (Title: {title}, Source: {source_info}, Scores: {score_info}) ---\n{content}")
            
        return "\n\n".join(chunks)
        
    except Exception as e:
        return f"RAG error: {e}"


def generate_rag_response(user_query: str, history: List[Dict[str, str]], context: str) -> str:
    # ... (کد این تابع بدون تغییر باقی می‌ماند) ...
    is_rag_active_and_valid = not context.startswith("RAG disabled:") and not context.startswith("RAG error:")
    is_context_useful = is_rag_active_and_valid and len(context.strip()) > 50

    base_sys_prompt = (
        "شما یک دستیار RAG هستید که به زبان فارسی مسلط است. "
        "همیشه مؤدب و حرفه‌ای پاسخ دهید. "
        "شما باید همیشه به **تاریخچه مکالمه** توجه کنید تا بتوانید سؤالات بعدی (مانانند 'حالا در مورد آن بگو') را پاسخ دهید. "
    )

    if is_context_useful:
        sys_prompt = (
            f"{base_sys_prompt} هدف اصلی شما پاسخ به سؤال کاربر بر اساس **تنها** 'RAG CONTEXT' ارائه شده است. "
            "شما **باید** پاسخ خود را با عبارت: 'بر اساس منبع دانش،' شروع کنید اگر پاسخ را در Context پیدا کردید. "
            "اگر Context پاسخ را ندارد، باید به وضوح بیان کنید: 'متأسفانه، منبع دانش من شامل اطلاعات لازم برای پاسخ به این سؤال نیست.' "
            "از دانش عمومی خود استفاده نکنید مگر اینکه Context به وضوح اطلاعات نامرتبتی ارائه دهد یا خالی باشد."
            "\n\n--- RAG CONTEXT ---\n"
            f"{context}\n---"
        )
    else:
        sys_prompt = (
            f"{base_sys_prompt} 'RAG CONTEXT' کافی یا مرتبطی یافت نشد. "
            "شما باید سؤال کاربر را با استفاده از **دانش عمومی** و **تاریخچه مکالمه** خود پاسخ دهید. "
            "چون RAG فعال نیست، **هیچ اشاره‌ای** به منبع دانش (مانند 'بر اساس منبع دانش') یا Context در پاسخ خود نکنید. "
            "فقط مستقیماً به سؤال کاربر پاسخ دهید."
        )

    response_text = gemini_service.generate(sys_prompt, history, user_query)
    
    return response_text


# این برای حافظه 
def main_streamlit_app():
    
    st.title(" 📚 Cohere RAG Chatbot ")
    
    st.info(f"💡 **وضعیت:** حافظه مکالمه‌ای فعال، **Embedding:** {EMBED_MODEL} ({cohere_embed_service.dim} Dim), **Reranking:** {RERANK_MODEL} ({TOP_K_RETRIEVAL} -> **{TOP_K_RERANK}**).", icon="🧠")

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "سلام، من یک دستیار RAG هستم. لطفاً سؤال خود را بپرسید.", "timestamp": datetime.now().isoformat()} 
        ]
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if "timestamp" in msg:
                try:
                    dt_obj = datetime.fromisoformat(msg["timestamp"])
                    time_str = dt_obj.strftime("%Y/%m/%d - %H:%M:%S")
                    alignment = "right" if msg["role"] == "user" else "left"
                    st.markdown(f'<p style="color: grey; font-size: small; text-align: {alignment}; margin-bottom: 0px;">{time_str}</p>', unsafe_allow_html=True)
                except ValueError:
                    st.markdown('<p style="color: red; font-size: small;">زمان نامعتبر</p>', unsafe_allow_html=True)


    if prompt := st.chat_input("سؤال خود را اینجا بنویسید..."):
        current_time = datetime.now().isoformat()
        
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": current_time})
        with st.chat_message("user"):
            st.markdown(prompt)
            dt_obj = datetime.fromisoformat(current_time)
            time_str = dt_obj.strftime("%Y/%m/%d - %H:%M:%S")
            st.markdown(f'<p style="color: grey; font-size: small; text-align: right; margin-bottom: 0px;">{time_str}</p>', unsafe_allow_html=True)


        with st.chat_message("assistant"):
            response_placeholder = st.empty() 
            
            with st.spinner("⏳ در حال بازیابی دانش و تولید پاسخ..."):
                
                # 1. بازیابی Context (شامل 5 چانک مرتب شده)
                rag_context = retrieve_rag_context(prompt) 
                
                # 2. تولید پاسخ با Context کامل
                response = generate_rag_response(prompt, st.session_state.messages, rag_context)

                response_placeholder.markdown(response)
                
                assistant_time = datetime.now().isoformat()
                
                st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": assistant_time})
                
                dt_obj = datetime.fromisoformat(assistant_time)
                time_str = dt_obj.strftime("%Y/%m/%d - %H:%M:%S")
                st.markdown(f'<p style="color: grey; font-size: small; text-align: left; margin-bottom: 0px;">{time_str}</p>', unsafe_allow_html=True)


                with st.expander("📝 جزئیات پردازش و منبع دانش"):
                    
                    is_rag_active_and_valid = not rag_context.startswith("RAG disabled:") and not rag_context.startswith("RAG error:")
                    
                    st.markdown(f"**مدل LLM:** `{GEMINI_MODEL}`")
                    st.markdown(f"**مدل Embedding:** `{EMBED_MODEL}` (Dim: {cohere_embed_service.dim})")
                    st.markdown(f"**مدل Reranker:** `{RERANK_MODEL}`")
                    st.markdown(f"**بازیابی اولیه:** {TOP_K_RETRIEVAL} چانک | **بازیابی نهایی (Rerank):** **{TOP_K_RERANK}** چانک")
                    st.markdown("---")
                    
                    if "**بر اساس منبع دانش،**" in response and is_rag_active_and_valid:
                        st.info("✅ **وضعیت Agent:** Agent از منبع دانش استفاده کرده و پاسخ را تولید کرده است.", icon="📚")
                    elif is_rag_active_and_valid and "No relevant documentation found." in rag_context:
                        st.warning("⚠️ **وضعیت Agent:** Context بازیابی شد، اما هیچ سند مرتبطی در دیتابیس پیدا نشد. Agent از دانش عمومی استفاده کرد.")
                    elif is_rag_active_and_valid:
                        st.warning("⚠️ **وضعیت Agent:** منبع دانش بازیابی شد، اما Agent تصمیم گرفت از آن استفاده نکند. (یا Context کافی نبود)")
                    else:
                        error_message = rag_context.split(': ', 1)[-1]
                        st.error(f"❌ **وضعیت Agent:** بازیابی منبع دانش ناموفق بود. \n\n**دلیل:** {error_message}", icon="🚨")
                    
                    st.markdown("---")
                    st.markdown(f"**تعداد پیام‌های در حافظه (Memory):** `{len(st.session_state.messages)}`")
                    
                    
                    # 🚨🚨🚨 منطق نمایش فقط مرتبط‌ترین چانک (چانک 1)
                    if rag_context and is_rag_active_and_valid:
                        
                        # پیدا کردن شروع چانک 1
                        start_index = rag_context.find("--- Chunk 1 ")
                        # پیدا کردن پایان چانک 1 (قبل از شروع چانک 2 یا انتهای متن)
                        end_index_2 = rag_context.find("--- Chunk 2 ", start_index)
                        
                        if start_index != -1: # اگر چانک 1 پیدا شد
                            if end_index_2 == -1:
                                # اگر Context فقط شامل 1 چانک است
                                display_chunk = rag_context[start_index:].strip()
                            else:
                                # اگر بیش از 1 چانک است، فقط چانک 1 را تا قبل از شروع چانک 2 می‌گیریم
                                display_chunk = rag_context[start_index:end_index_2].strip()

                            st.text_area("📄 **مرتبط‌ترین منبع (Context):**", display_chunk, height=300)
                            
                            # نمایش Context کامل در یک بخش جداگانه (کوچکتر)
                            st.text_area("✨ **Context کامل (5 چانک) ارسالی به Gemini**", rag_context, height=100, help="این متنی است که Gemini برای تولید پاسخ شما از آن استفاده کرده است.")
                        else:
                            st.text_area("✨ **Context کامل (5 چانک) ارسالی به Gemini**", rag_context, height=100)
                    else:
                        st.markdown("**Context مرتبطی برای نمایش یافت نشد.**")

if __name__ == "__main__":
    cohere_embed_service, gemini_service = get_clients() 
    main_streamlit_app()
