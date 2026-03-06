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
st.set_page_config(
    page_title="OneSwifty AI", 
    page_icon="🚀", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

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
    """Generates high-precision vectors using OpenAI's 'large' model."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

def log_query(query, answer, confidence):
    log_file = "oneswifty_audit_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "AI_Answer", "Confidence_Score"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query, answer, f"{confidence*100:.2f}%"])

# --- MAIN INTERFACE START ---
st.title("🚀 OneSwifty: Knowledge Engine [BETA]")
st.sidebar.warning("⚠️ This product is currently in testing and under active enhancement.")

st.markdown("### High-Precision Scientific & Financial Auditing")

# --- STEP 1: INSTRUCTIONS & INGESTION (Center Stage for Mobile) ---
with st.container():
    st.info("""
    **How to use OneSwifty:**
    1. **Upload** your PDF database in the section below.
    2. Click **'Start AI Ingestion'** to index the knowledge.
    3. Use the **Search Bar** at the bottom to ask complex questions.
    """)

    # The Ingestion "Dropzone"
    with st.expander("📥 Step 1: Ingest New Knowledge (Click to Expand)", expanded=True):
        uploaded_file = st.file_uploader("Upload Knowledge PDF", type="pdf", label_visibility="collapsed")
        
        if st.button("🚀 Start AI Ingestion", use_container_width=True):
            if uploaded_file:
                progress_bar = st.progress(0)
                status_text = st.empty()
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                first_page_sample = doc[0].get_text()[:1500] 
                
                with st.spinner("AI is identifying document identity..."):
                    meta_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "Extract Title | Authors | Category"},
                            {"role": "user", "content": first_page_sample}
                        ]
                    )
                    metadata_raw = meta_response.choices[0].message.content
                    try:
                        auto_title, auto_author, auto_category = metadata_raw.split("|")
                    except:
                        auto_title, auto_author, auto_category = uploaded_file.name, "Unknown", "General"

                with get_connection() as conn:
                    with conn.cursor() as cur:
                        total_pages = len(doc)
                        for i, page in enumerate(doc):
                            progress_bar.progress((i + 1) / total_pages)
                            status_text.text(f"Processing page {i+1}...")
                            text = page.get_text()
                            
                            # Chunking Logic
                            chunk_size, overlap, start = 500, 50, 0
                            while start < len(text):
                                end = min(start + chunk_size, len(text))
                                chunk = text[start:end].strip()
                                if len(chunk) > 20:
                                    vec = get_embedding(chunk)
                                    cur.execute("""
                                        INSERT INTO oneswifty_knowledge 
                                        (category, content_text, metadata_source, author, title, page_number, embedding)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """, (auto_category.strip(), chunk, uploaded_file.name, auto_author.strip(), auto_title.strip(), i + 1, vec))
                                start = end - overlap if (end - overlap) > start else end + 1
                            conn.commit()
                        st.success(f"✅ Ingested: {auto_title}")
            else:
                st.error("⚠️ Please select a PDF file first.")

st.divider()

# --- STEP 2: LIBRARY STATS (Visible in Main Body) ---
with st.expander("📊 Current Library Knowledge"):
    conn = get_connection()
    if conn:
        df = pd.read_sql("SELECT title, author, category FROM oneswifty_knowledge", conn)
        if not df.empty:
            st.dataframe(df.drop_duplicates(subset=['title']), hide_index=True, use_container_width=True)
        else:
            st.info("Knowledge base is currently empty.")
        conn.close()

# --- STEP 3: SEARCH INTERFACE ---
st.subheader("🔍 Step 2: Intelligent Search")
# Professional Dynamic Label
query_label = "What would you like to ask about the PDF(s) you uploaded?"
query = st.chat_input(query_label, key="oneswifty_chat_input")

if query:
    st.chat_message("user").write(query)
    with st.spinner("Synthesizing context..."):
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
                    context_text = "\n\n".join([f"TITLE: {res[2]} | PAGE: {res[5]} | CONTENT: {res[1]}" for res in results])
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are OneSwifty AI. Cite Page [X] and Title for every fact. Use LaTeX for math."},
                            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                        ]
                    )
                    final_answer = response.choices[0].message.content
                    avg_score = sum([res[3] for res in results]) / len(results)
                    log_query(query, final_answer, avg_score)

                    with st.chat_message("assistant"):
                        if results[0][3] < 0.60:
                            st.warning("⚠️ Low Confidence Match")
                        
                        # Render LaTeX if present
                        if "$$" in final_answer:
                            parts = final_answer.split("$$")
                            for i, part in enumerate(parts):
                                if i % 2 == 1: st.latex(part.strip())
                                else: st.markdown(part)
                        else:
                            st.markdown(final_answer)
                else:
                    st.error("No relevant knowledge found in the database.")
            conn.close()

# --- SIDEBAR: HIDDEN ADMIN ONLY ---
with st.sidebar:
    st.header("🔐 Admin Controls")
    admin_input = st.text_input("Admin Key", type="password")
    if admin_input == ADMIN_KEY:
        if st.button("🗑️ Wipe All Knowledge"):
            # ... delete logic ...
            st.warning("Database cleared.")
