import streamlit as st
import os
import psycopg
import fitz  # PyMuPDF
import csv
import pandas as pd
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

# Load environment variables
load_dotenv()

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Admin Secret ---
ADMIN_KEY = "Swifty2026" 

# --- Page Config ---
st.set_page_config(page_title="OneSwifty AI", page_icon="🚀", layout="wide")
st.title("🚀 OneSwifty: Universal Knowledge Engine")

# --- Database Connection ---
def get_connection():
    try:
        conn = psycopg.connect(
            host=st.secrets["DB_HOST"],
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASSWORD"],
            port=st.secrets["DB_PORT"],
            prepare_threshold=None
            
        )
        register_vector(conn)
        return conn
    except Exception as e:
        st.error(f"❌ Database Connection Error: {e}")
        return None

def get_embedding(text):
    """Generates high-precision vectors using OpenAI's 'large' model (3,072 dims)."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# --- Logging Function ---
def log_query(query, answer, confidence):
    """Saves interaction to a local CSV for performance auditing."""
    log_file = "oneswifty_audit_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "AI_Answer", "Confidence_Score"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            query, 
            answer, 
            f"{confidence*100:.2f}%"
        ])


    
    # RESTORED: Show Library Stats (Publicly viewable in sidebar)
    st.header("📊 Library Stats")
    conn = get_connection()
    if conn:
        df = pd.read_sql("SELECT title, author, category FROM oneswifty_knowledge", conn)
        if not df.empty:
            st.dataframe(df.drop_duplicates(subset=['title']), hide_index=True)
        else:
            st.info("Knowledge base is currently empty.")
        conn.close()

    st.divider()
    
    # Admin Access Controls
    st.header("🔐 Admin Controls")
    admin_input = st.text_input("Admin Key", type="password")
    
    if admin_input == ADMIN_KEY:
        if st.button("🗑️ Clear All Knowledge"):
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM oneswifty_knowledge")
                    conn.commit()
                st.warning("Knowledge Base Wiped.")
                conn.close()
                st.rerun()
        
        if os.path.isfile("oneswifty_audit_log.csv"):
            with open("oneswifty_audit_log.csv", "rb") as file:
                st.download_button(label="📥 Download Audit Log", data=file, file_name="oneswifty_audit_log.csv", mime="text/csv")

# --- Main Interface: Semantic Search ---
uploaded_file = st.file_uploader("Upload Knowledge PDF", type="pdf")
    
if st.button("Ingest to Database"):
        if uploaded_file:
            progress_bar = st.progress(0)
            status_text = st.empty()
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            first_page_sample = doc[0].get_text()[:1500] 
            
            with st.spinner("AI is identifying document identity..."):
                meta_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a technical librarian. Extract the ACTUAL FORMAL TITLE, the FULL LIST of all authors (comma-separated), and a one-word CATEGORY. Return: Title | Author1, Author2, etc. | Category"},
                        {"role": "user", "content": first_page_sample}
                    ]
                )
                metadata_raw = meta_response.choices[0].message.content
                try:
                    auto_title, auto_author, auto_category = metadata_raw.split("|")
                    auto_title, auto_author, auto_category = auto_title.strip(), auto_author.strip(), auto_category.strip()
                except ValueError:
                    auto_title, auto_author, auto_category = uploaded_file.name, "Unknown", "General"

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SET statement_timeout = '300s';")
                    total_pages = len(doc)
                    for i, page in enumerate(doc):
                        progress_bar.progress((i + 1) / total_pages)
                        status_text.text(f"Processing page {i+1} of {total_pages}...")
                        text = page.get_text()
                        
                        chunk_size, overlap, start = 500, 50, 0
                        while start < len(text):
                            end = min(start + chunk_size, len(text))
                            if end < len(text):
                                last_space = text.rfind(' ', start, end)
                                if last_space != -1: end = last_space
                            chunk = text[start:end].strip()
                            if len(chunk) > 20:
                                clean_chunk = "".join(char for char in chunk if char != "\x00").strip()
                                vec = get_embedding(clean_chunk)
                                cur.execute("""
                                    INSERT INTO oneswifty_knowledge 
                                    (category, content_text, metadata_source, author, title, page_number, embedding)
                                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                                """, (auto_category, clean_chunk, uploaded_file.name, auto_author, auto_title, i + 1, vec))
                            start = end - overlap if (end - overlap) > start else end + 1
                        conn.commit()
                    st.success(f"✅ Ingested: {auto_title}")
        else: st.error("⚠️ Upload a PDF.")

st.divider()
    
    # RESTORED: Show Library Stats (Publicly viewable in sidebar)
st.header("📊 Library Stats")
conn = get_connection()
if conn:
        df = pd.read_sql("SELECT title, author, category FROM oneswifty_knowledge", conn)
        if not df.empty:
            st.dataframe(df.drop_duplicates(subset=['title']), hide_index=True)
        else:
            st.info("Knowledge base is currently empty.")
        conn.close()

st.divider()
    
st.subheader("🔍 Intelligent Search")
query = st.chat_input("What would you like to know?", key="oneswifty_chat_input")

if query:
     st.chat_message("user").write(query)
with st.spinner("Synthesizing context from across document pages..."):
        query_vec = get_embedding(query)
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT category, content_text, title, 1 - (embedding <=> %s::vector) AS similarity, author, page_number
                    FROM oneswifty_knowledge 
                    ORDER BY similarity DESC LIMIT 5
                """, (query_vec,))
                results = cur.fetchall()
                
                if results:
                    context_text = "\n\n".join([f"TITLE: {res[2]} | AUTHORS: {res[4]} | PAGE: {res[5]} | CONTENT: {res[1]}" for res in results])
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system", 
"content": """You are OneSwifty AI, a high-precision Scientific and Financial Auditor. 

MANDATORY CITATION RULE:
Every factual statement MUST be cited using the format: 
'As seen on Page [X] in [Full Paper Title] by [Primary Author] et al...'

FINANCIAL & TABLE AUDIT RULES:
1. HIERARCHY: When asked for a category total, do NOT grab the first number you see. Look specifically for a line that contains the word 'Total' (e.g., 'Insurance and other financial reserves Total').
2. VERIFICATION: If you list sub-items, verify their sum against the reported 'Total' line in the context. If they differ, state that you are reporting the explicit 'Total' line from the document.
3. CURRENCY: All figures in the Budget documents are in MILLIONS of dollars unless otherwise stated. Convert $788,871 to $788.9 Billion in your final explanation for readability.

MULTI-PAGE SYNTHESIS:
1. Connect definitions (e.g., Page 11) to applications (e.g., Page 21) across different chunks.
2. For Equation 3.22, provide the LaTeX: $$max_{0\\le z\\le z_{in}}f_{MG}(z)>1$$
"""
                            },
                            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                        ]
                    )
                    final_answer = response.choices[0].message.content
                    best_score = results[0][3]
                    avg_score = sum([res[3] for res in results]) / len(results)
                    log_query(query, final_answer, avg_score)

                    with st.chat_message("assistant"):
                        if best_score < 0.59:
                            st.warning(f"⚠️ **Low Confidence Match ({best_score*100:.1f}%)**")
                        
                        if "$$" in final_answer:
                            parts = final_answer.split("$$")
                            for i, part in enumerate(parts):
                                if i % 2 == 1: st.success("📝 **Technical Formula Found:**"); st.latex(part.strip())
                                else: st.markdown(part)
                        else:
                            st.markdown(final_answer)
                        
                        with st.expander("🔍 Scientific Audit Logs"):
                            st.write(f"**Primary Match Score:** {best_score*100:.2f}%")
                            for res in results:
                                st.caption(f"**Doc:** {res[2]} | **Page:** {res[5]} | **Relevance:** {res[3]*100:.1f}%")
                else:
                    st.error("No knowledge found.")
                conn.close()
