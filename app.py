import streamlit as st
import os
import psycopg
import fitz  # PyMuPDF
import csv
import smtplib
import pandas as pd
from datetime import datetime
from email.mime.text import MIMEText
from openai import OpenAI
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

# Load environment variables
load_dotenv()

# --- Initialize OpenAI Client ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- CONFIGURATION ---
ADMIN_KEY = "Swifty2026" 
DAILY_BUDGET_LIMIT = 4.50
ALERT_THRESHOLD = 0.80  # 80%
USER_EMAIL = "rimaaleryani@gmail.com"

# --- Page Config ---
st.set_page_config(page_title="OneSwifty AI", page_icon="🚀", layout="wide", initial_sidebar_state="collapsed")

# --- Database & Utility Functions ---
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
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return response.data[0].embedding

def send_budget_alert(current_spend):
    sender_email = st.secrets["EMAIL_USER"]
    password = st.secrets["EMAIL_PASSWORD"] 
    msg = MIMEText(f"OneSwifty Alert: You have spent ${current_spend:.2f}, which is 80% of your daily limit.")
    msg['Subject'] = "🚀 OneSwifty Budget Alert"
    msg['From'] = sender_email
    msg['To'] = USER_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, USER_EMAIL, msg.as_string())
    except Exception as e:
        print(f"Email failed: {e}")

def get_total_spend_today():
    log_file = "oneswifty_audit_log.csv"
    if not os.path.isfile(log_file): return 0.0
    try:
        df = pd.read_csv(log_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        today_df = df[df['Timestamp'].dt.date == datetime.now().date()]
        return today_df['Cost_USD'].astype(float).sum()
    except: return 0.0

def log_query(query, answer, confidence, in_t, out_t):
    log_file = "oneswifty_audit_log.csv"
    cost_usd = (in_t * 0.0000025) + (out_t * 0.000010)
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Confidence", "In_Tokens", "Out_Tokens", "Cost_USD"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), query[:100], f"{confidence*100:.2f}%", in_t, out_t, f"{cost_usd:.6f}"])

# --- MAIN INTERFACE ---
st.title("🚀 OneSwifty: Knowledge Engine [BETA]")

# --- INSTRUCTIONS ---
st.markdown("""
### 📖 How to use OneSwifty:
1.  **Ingest Knowledge**: Use the expander below to upload a PDF. This trains the AI on your specific document.
2.  **Verify Library**: Check the table in Step 2 to ensure your document was processed correctly.
3.  **Search & Audit**: Use the search bar at the bottom to ask technical questions. OneSwifty will cite specific pages and titles for every answer.
---
""")

current_spend = get_total_spend_today()
is_over_budget = current_spend >= DAILY_BUDGET_LIMIT

# Budget Alerts Logic
if current_spend >= (DAILY_BUDGET_LIMIT * ALERT_THRESHOLD) and not is_over_budget:
    if 'alert_sent' not in st.session_state:
        send_budget_alert(current_spend)
        st.session_state.alert_sent = True
    st.warning(f"⚠️ Budget Warning: You have used ${current_spend:.2f} (80% of limit).")

# --- STEP 1: INGESTION ---
with st.container():
    with st.expander("📥 Step 1: Ingest New Knowledge", expanded=not is_over_budget):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        if st.button("🚀 Start AI Ingestion", use_container_width=True, disabled=is_over_budget):
            if uploaded_file:
                # Initialize UI elements for feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                total_pages = len(doc)
                first_page_sample = doc[0].get_text()[:2000]
                
                with st.spinner("AI analyzing document identity..."):
                    meta_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a technical librarian. Extract Title | Institutional Author | Category. Ignore file names."},
                            {"role": "user", "content": first_page_sample}
                        ]
                    )
                    try:
                        res_text = meta_response.choices[0].message.content
                        auto_title, auto_author, auto_category = res_text.split("|")
                    except:
                        auto_title, auto_author, auto_category = uploaded_file.name, "Unknown", "General"
                
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        for i, page in enumerate(doc):
                            # Update Progress
                            progress_bar.progress((i + 1) / total_pages)
                            status_text.text(f"Processing page {i+1} of {total_pages}...")
                            
                            text = page.get_text()
                            if len(text.strip()) > 50:
                                # Chunking logic
                                chunk_size, overlap, start = 1000, 100, 0
                                while start < len(text):
                                    end = min(start + chunk_size, len(text))
                                    
                                    raw_chunk = text[start:end].strip()
                                    clean_chunk = raw_chunk.replace("\x00", "").strip()
                                if len(clean_chunk) > 20:
                                    # Generate embedding from the clean text
                                    vec = get_embedding(clean_chunk)
                                  
                                    cur.execute("""
                                        INSERT INTO oneswifty_knowledge 
                                        (category, content_text, metadata_source, author, title, page_number, embedding)
                                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                                    """, (auto_category.strip(), chunk, uploaded_file.name, auto_author.strip(), auto_title.strip(), i + 1, vec))
                                    
                                    start = end - overlap if (end - overlap) > start else end + 1
                        conn.commit()
                st.success(f"✅ Ingested: {auto_title}")
            else:
                st.error("⚠️ Please upload a file first.")

st.divider()

# --- STEP 2: LIBRARY KNOWLEDGE ---
st.subheader("📊 Step 2: Current Library Knowledge")
conn = get_connection()
if conn:
    try:
        df = pd.read_sql("SELECT title, author, category FROM oneswifty_knowledge", conn)
        if not df.empty:
            st.dataframe(df.drop_duplicates(subset=['title']), hide_index=True, use_container_width=True)
        else:
            st.info("The knowledge library is currently empty.")
    except:
        st.info("Waiting for first ingestion...")
    conn.close()

st.divider()

# --- STEP 3: SEARCH ---
if is_over_budget:
    st.error(f"🛑 Daily Budget Reached (${DAILY_BUDGET_LIMIT}). Search is disabled until tomorrow.")
else:
    st.subheader("🔍 Step 3: Intelligent Search")
    query = st.chat_input("What would you like to ask about the uploaded files?")
    if query:
        st.chat_message("user").write(query)
        with st.spinner("OneSwifty is auditing documents..."):
            query_vec = get_embedding(query)
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""SELECT content_text, title, 1 - (embedding <=> %s::vector) AS sim, page_number 
                                FROM oneswifty_knowledge ORDER BY sim DESC LIMIT 5""", (query_vec,))
                    results = cur.fetchall()
                    if results:
                        context = "\n".join([f"Source: {r[1]} P.{r[3]}: {r[0]}" for r in results])
                        resp = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "system", "content": "You are OneSwifty AI. Cite Page and Title for facts."},
                                      {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}]
                        )
                        st.chat_message("assistant").write(resp.choices[0].message.content)
                        log_query(query, "...", 0.9, resp.usage.prompt_tokens, resp.usage.completion_tokens)
                conn.close()

# --- SIDEBAR ADMIN ---
with st.sidebar:
    st.header("🔐 Admin Controls")
    admin_input = st.text_input("Admin Key", type="password")
    if admin_input == ADMIN_KEY:
        st.metric("Today's Spend", f"${current_spend:.4f}")
        st.progress(min(current_spend / DAILY_BUDGET_LIMIT, 1.0))
        if st.button("🗑️ Wipe Database"):
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM oneswifty_knowledge")
                    conn.commit()
                st.warning("Database Cleared.")
                conn.close()
                st.rerun()
