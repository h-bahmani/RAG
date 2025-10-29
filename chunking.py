
import os
import asyncio
import json
import random 
import re 
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import httpx 
from docx import Document
# 🚨 افزودن Table
from docx.table import Table
from docx.text.paragraph import Paragraph
from dotenv import load_dotenv, find_dotenv

# --- کتابخانه‌های ضروری ---
try:
    from supabase import create_client, Client
    import cohere 
except ImportError:
    print("🛑 Critical: Ensure you have 'pip install supabase cohere python-dotenv python-docx'")
    sys.exit(1)

# ----------------------------------------
# ۱. تنظیمات و متغیرهای محیطی
# ----------------------------------------
load_dotenv(find_dotenv(), override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# 👇 تنظیمات COHERE
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "") 
COHERE_EMBED_MODEL = "embed-multilingual-v3.0" 

# 🚨 ابعاد نهایی تنظیم شده در Supabase - باید 1024 باشد!
FALLBACK_EMBED_DIM = 1024 

# 💡 تنظیمات Semantic Chunking
MAX_CHUNK_SIZE = 700 
CHUNK_OVERLAP = 150 

# ----------------------------------------
# ۲. نمونه‌سازی کلاینت‌ها و بررسی تنظیمات
# ----------------------------------------
if not all([SUPABASE_URL, SUPABASE_SERVICE_KEY, COHERE_API_KEY]): 
    print("🛑 Critical: API Keys (COHERE_API_KEY) and Supabase settings must be set.")
    sys.exit(1)
    
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("✅ Supabase client initialized.")
except Exception as e:
    print(f"❌ Configuration Error: Could not initialize Supabase client: {e}")
    sys.exit(1)

# ----------------------------------------
# ۳. کلاس‌های سرویس LLM (با کلاینت COHERE و اصلاح متد close)
# ----------------------------------------

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] 

class CohereEmbedClient:
    """کلاس برای مدیریت تماس‌های ناهمگام و مستقیم با Cohere REST API."""
    
    def __init__(self, api_key: str, embed_model: str, dim: int):
        self.api_key = api_key
        self.embed_model = embed_model
        self.dim = dim
        # 🚨 ایجاد AsyncClient
        self.co = cohere.AsyncClient(api_key=api_key) 
        
    async def embed_content_direct(self, text: str) -> Optional[List[float]]:
        """دریافت Embedding با Cohere."""
        
        MAX_RETRIES = 3 
        base_delay = 5 
        
        for attempt in range(MAX_RETRIES):
            try:
                # 🚨 Cohere Client V1 نیاز به این پارامترها دارد
                response = await self.co.embed(
                    texts=[text],
                    model=self.embed_model,
                    input_type="search_document" 
                )
                
                vector = response.embeddings[0]
                
                if len(vector) != self.dim:
                    print(f"❌ Cohere returned {len(vector)} dims, expected {self.dim}. Skipping.")
                    return None
                
                return vector
                    
            except cohere.errors.CohereAPIError as e:
                if e.http_status in [429, 500, 503] and attempt < MAX_RETRIES - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1) 
                    print(f"   ⚠️ Cohere API Server Error ({e.http_status}). Retrying in {delay:.2f}s (Attempt {attempt + 1}/{MAX_RETRIES}).")
                    await asyncio.sleep(delay)
                    continue 
                
                print(f"❌ Cohere API Error {e.http_status}. Details: {e.message}")
                return None
            
            except Exception as e:
                print(f"❌ General Error during API call: {type(e).__name__}: {e}.")
                return None
        
        return None 
        
    async def close(self):
        """🚨 رفع مشکل AttributeError: 'AsyncClient' object has no attribute 'close'"""
        # Cohere V1 (AsyncClient) از aclose پشتیبانی می‌کند
        if hasattr(self.co, 'aclose'):
            await self.co.aclose()
        # در غیر این صورت، کاری انجام نمی‌دهیم تا خطا ندهد.
        elif hasattr(self.co, 'close'):
            # اگرچه AsyncClient این را ندارد، اما به عنوان یک Fallback می‌ماند.
            self.co.close()
        else:
            print("⚠️ Warning: Cohere AsyncClient does not have a close/aclose method.")

# نمونه‌سازی نهایی کلاینت
cohere_embed_client = CohereEmbedClient( 
    COHERE_API_KEY, 
    COHERE_EMBED_MODEL,
    FALLBACK_EMBED_DIM
)

# ----------------------------------------
# ۴. توابع ابزاری و هسته RAG
# ----------------------------------------

def simple_text_heuristic(chunk: str) -> Dict[str, str]:
    # ... (بدون تغییر) ...
    lines = [l.strip() for l in chunk.splitlines() if l.strip()]
    
    if lines:
        title = lines[0].lstrip('#*->- ').strip()[:80]
        
        sentences = re.split(r'(?<=[.!?])\s+', " ".join(lines))
        summary = " ".join([s.strip() for s in sentences[:3] if s.strip()])[:300]
        
        if not summary and lines:
            summary = " ".join(lines[:2])[:300]

    else:
        title = "Untitled (Empty Chunk)"
        summary = "No content to summarize."
        
    return {"title": title, "summary": summary.strip()}


def chunk_text_by_sentence(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # ... (بدون تغییر) ...
    sentences = re.split(r'(?<=[.?!])\s+', text)
    if not sentences:
        return []

    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        if len(current_chunk) + len(sentence) + 1 > max_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            overlap_sentences = []
            overlap_len = 0
            
            temp_sentences = re.split(r'(?<=[.?!])\s+', current_chunk)
            
            for s in reversed(temp_sentences):
                s = s.strip()
                if not s: continue
                
                if overlap_len + len(s) + 1 <= overlap:
                    overlap_sentences.insert(0, s) 
                    overlap_len += len(s) + 1
                else:
                    break
            
            current_chunk = " ".join(overlap_sentences).strip()
            
            current_chunk = (current_chunk + " " + sentence).strip()
            
        else:
            current_chunk = (current_chunk + " " + sentence).strip()

    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks


async def insert_chunk(chunk: ProcessedChunk):
    
    if not supabase: return None
    
    if chunk.embedding is None or len(chunk.embedding) != FALLBACK_EMBED_DIM: 
        print(f"🛑 Skipping insertion for chunk {chunk.chunk_number} (Invalid embedding length: {len(chunk.embedding) if chunk.embedding else 0}).")
        return None
        
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        
        print(f"✅ Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        if 'duplicate key value violates unique constraint' in str(e):
             print(f"⚠️ Insertion Skipped: Chunk {chunk.chunk_number} for {chunk.url} already exists in DB.")
        else:
             print(f"❌ Error inserting chunk: {e}") 
        return None

async def process_chunk(chunk: str, chunk_number: int, url: str) -> Optional[ProcessedChunk]:
    
    fallback_data = simple_text_heuristic(chunk)
    title = fallback_data["title"]
    summary = fallback_data["summary"]
    llm_summary_used = False 

    meta_chunk = f"Title: {title}\nSummary: {summary}\n\nCONTENT:\n{chunk}"

    embedding = await cohere_embed_client.embed_content_direct(meta_chunk)
    
    if embedding is None or len(embedding) != FALLBACK_EMBED_DIM:
        print(f"🛑 CRITICAL: Embedding failed or had wrong dimension for chunk {chunk_number}. Skipping insertion.")
        return None 
        
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=title,
        summary=summary,
        content=chunk,
        embedding=embedding,
        metadata={
            "source": url,
            "chunk_length": len(chunk),
            "llm_summary_used": llm_summary_used,
            "embedding_model": COHERE_EMBED_MODEL, 
            "retrieved_at": datetime.now(timezone.utc).isoformat()
        }
    )


def extract_table_text(table: Table) -> str:
    """جدول را به فرمت متنی قابل خواندن تبدیل می‌کند."""
    table_data = []
    
    # 🚨 ایجاد هدر برای خوانایی بهتر جدول (مثلاً: Table X: ...)
    table_data.append("--- TABLE START ---")
    
    for i, row in enumerate(table.rows):
        row_text = []
        for j, cell in enumerate(row.cells):
            # تمیز کردن متن هر سلول
            cell_content = cell.text.replace('\n', ' ').strip()
            # فرمت: [Column 1: Value] [Column 2: Value]
            row_text.append(f"[C{j+1}: {cell_content}]")
            
        table_data.append(f"Row {i+1}: {' '.join(row_text)}")
        
    table_data.append("--- TABLE END ---")
    return '\n'.join(table_data)

def extract_text_from_docx(file_path: str) -> str:
    """Extracts all text content (including tables) from a .docx file."""
    try:
        document = Document(file_path)
        full_content = []
        
        for element in document.element.body:
            # 🚨 پردازش پاراگراف
            if element.tag.endswith('p'): # Paragraph
                paragraph = Paragraph(element, document)
                text = paragraph.text.strip()
                if text:
                    full_content.append(text)
            
            # 🚨 پردازش جدول
            elif element.tag.endswith('tbl'): # Table
                table = Table(element, document)
                table_text = extract_table_text(table)
                if table_text:
                    full_content.append('\n' + table_text + '\n')
                    
        return '\n\n'.join(full_content)
    except Exception as e:
        print(f"❌ Error extracting text from DOCX: {e}")
        return ""


async def process_and_store_document(url: str, content: str):
    
    chunks = chunk_text_by_sentence(content) 
    print(f"Divided document into {len(chunks)} semantic chunks (Max Size: {MAX_CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")
    
    processed_chunks = []
    
    for i, chunk in enumerate(chunks):
        
        if not chunk.strip():
            print(f"⚠️ Skipping empty chunk {i}.")
            continue
            
        print(f"\n⚙️ Processing Chunk {i} of {len(chunks)} (Length: {len(chunk)})...")
        
        try:
            processed_chunk = await process_chunk(chunk, i, url)
            
            if processed_chunk is not None:
                processed_chunks.append(processed_chunk)
                
            if processed_chunk is not None and i < len(chunks) - 1:
                delay = 5 + random.uniform(0, 1) 
                print(f"⏳ Waiting {delay:.2f} seconds after successful embedding (Rate Limit Avoidance)...")
                await asyncio.sleep(delay)
                
        except Exception as e:
            print(f"❌ Critical error during chunk processing {i}: {type(e).__name__}: {e}. Skipping chunk.")
            
    
    print("\nStarting SERIAL insertion to Supabase...")
    for chunk in processed_chunks:
        await insert_chunk(chunk)
    


async def process_local_file(file_path: str):
    # ... (بدون تغییر) ...
    content = ""
    source_name = os.path.basename(file_path)
    
    # 🚨 بهبود یافته برای استخراج جدول از docx
    if file_path.lower().endswith('.docx'):
        print(f"📄 Processing DOCX: {source_name}")
        content = extract_text_from_docx(file_path)
    elif file_path.lower().endswith(('.txt', '.md')):
          print(f"📄 Processing Text/MD: {source_name}")
          try:
              with open(file_path, "r", encoding="utf-8") as f:
                  content = f.read()
          except Exception as e:
              print(f"❌ Error reading file: {e}")

    content_length = len(content.strip())
    print(f"💡 Extracted content length: {content_length} characters.")
    
    if not content or content_length == 0:
        print(f"🛑 Process stopped: Extracted content from {source_name} is empty or extraction failed.")
        return

    source_url = f"local://{source_name}"
    await process_and_store_document(source_url, content)
    
    print(f"\n✅ Finished RAG pipeline for local file: {file_path}")

# ----------------------------------------
# ۵. تابع اصلی
# ----------------------------------------
async def main():
    
    # 🚨 مسیر فایل را بررسی کنید.
    local_document_file = r"C:\Users\KIMI\Desktop\n8n-local\crawl4AI-agent\doc\New Microsoft Word Document.docx"
    
    if not os.path.exists(local_document_file):
        print(f"❌ Document file not found at: {local_document_file}")
        return
    
    print(f"🚀 Starting RAG Pipeline | Embed Model: {COHERE_EMBED_MODEL} ({FALLBACK_EMBED_DIM} Dim)")
    print("-----------------------------------------------------------------")
    
    if not COHERE_API_KEY or len(COHERE_API_KEY) < 30:
        print("\n🛑 FATAL ERROR: COHERE_API_KEY is missing or too short. Check your .env file.")
        return

    try:
        await process_local_file(local_document_file)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed during execution. **Detailed Error:** {type(e).__name__}: {e}")
        
    finally:
        print("\n🧹 Closing clients...")
        # 🚨 فراخوانی متد close اصلاح شده
        await cohere_embed_client.close()


if __name__ == "__main__":
    
    try:
        # 🚨 مطمئن شوید که asyncio.run فقط یک بار فراخوانی می‌شود
        asyncio.run(main())
    except ValueError as e:
        print(f"❌ Initialization Error: {e}")
