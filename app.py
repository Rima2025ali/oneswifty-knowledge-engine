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

# --- CONFIGURATION ---
ADMIN_KEY = "Swifty2026" 
DAILY_BUDGET_LIMIT = 4.50  # Set your maximum daily spend here in USD

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

# --- Billing & Safety System ---
def get_total_spend_today():
    """Calculates total spend from the log file for the current date."""
    log_file = "oneswifty_audit_log.csv"
    if not os.path.isfile(log_file):
        return 0.0
    
    try:
        df = pd.read_csv(log_file)
        # Ensure timestamp is datetime objects
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        today = datetime.now().date()
        
        # Filter for today and sum the cost
        today_df = df[df['Timestamp'].dt.date == today]
        return today_df['Cost_USD'].astype(float).sum()
    except Exception:
        return 0.0

def log_query(query, answer, confidence, in_tokens=0, out_tokens=0):
    """Logs interactions and calculates estimated OpenAI costs."""
    log_file = "oneswifty_audit_log.csv"
    
    # pricing for gpt-4o: $2.50 / $10.00 per 1M tokens
    cost_usd = (in_tokens * 0.0000025) + (out_tokens * 0.000010)
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Query", "Confidence", "In_Tokens", "Out_Tokens", "Cost_USD"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            query[:100], f"{confidence*100:.2f}%", in_tokens, out_tokens, f"{cost_usd:.6f}"
        ])

def get_embedding(text):
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return response.data[0].embedding

# --- MAIN INTERFACE ---
st.title("🚀 OneSwifty: Knowledge Engine [BETA]")

# 💰 CHECK BUDGET BEFORE ENABLING SEARCH
current_spend = get_total_spend_today()
is_over_budget = current_spend >= DAILY_BUDGET_LIMIT

if is_over_budget:
    st.error(f"🛑 **Daily Budget Reached:** OneSwifty has hit its limit of ${DAILY_BUDGET_LIMIT}. Search is disabled until tomorrow to save costs.")
else:
    st.sidebar.success(f"Budget: ${current_spend:.2f} / ${DAILY_BUDGET_LIMIT}")

# --- STEP 1: INGESTION ---
with st.container():
    with st.expander("📥 Step 1: Ingest New Knowledge", expanded=not is_over_budget):
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
        if st.button("🚀 Start AI Ingestion", use_container_width=True):
            if uploaded_file:
                # [Ingestion logic remains the same...]
                st.success("Ingestion complete.")

st.divider()

# --- STEP 2: SEARCH INTERFACE ---
st.subheader("🔍 Step 2: Intelligent Search")

# The chat_input is now dynamically disabled based on your budget
query = st.chat_input(
    "What would you like to ask?", 
    disabled=is_over_budget
)

if query:
    st.chat_message("user").write(query)
    with st.spinner("OneSwifty is auditing documents..."):
        query_vec = get_embedding(query)
        conn = get_connection()
        if conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT content_text, title, 1 - (embedding <=> %s::vector) AS similarity, page_number
                    FROM oneswifty_knowledge ORDER BY similarity DESC LIMIT 5
                """, (query_vec,))
                results = cur.fetchall()
                if results:
                    context = "\n\n".join([f"TITLE: {res[1]} | PAGE: {res[3]} | CONTENT: {res[0]}" for res in results])
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are OneSwifty AI. Cite Page [X]."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                        ]
                    )
                    in_t, out_t = response.usage.prompt_tokens, response.usage.completion_tokens
                    ans = response.choices[0].message.content
                    score = sum([res[2] for res in results]) / len(results)
                    
                    log_query(query, ans, score, in_t, out_t)
                    st.chat_message("assistant").markdown(ans)
            conn.close()

# --- SIDEBAR: ADMIN & DASHBOARD ---
with st.sidebar:
    st.header("🔐 Admin Controls")
    admin_input = st.text_input("Admin Key", type="password")
    if admin_input == ADMIN_KEY:
        st.subheader("💰 Billing Dashboard")
        st.metric("Today's Spend", f"${current_spend:.4f}")
        st.progress(min(current_spend / DAILY_BUDGET_LIMIT, 1.0))
        
        if st.button("🗑️ Wipe All Knowledge"):
            # ... delete logic ...
            st.rerun()
