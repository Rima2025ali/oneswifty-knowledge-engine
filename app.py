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
import re

# Load environment variables
load_dotenv()

# --- Initialize OpenAI Client ---
try:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception as e:
    st.error("❌ OpenAI API Key missing or invalid in Secrets dashboard.")

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
    """Generates high-precision vectors using OpenAI's 'large' model (3,072 dims)."""
    if not text: return None
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return response.data[0].embedding

def send_budget_alert(current_spend):
    sender_email = st.secrets.get("EMAIL_USER")
    password = st.secrets.get("EMAIL_PASSWORD") 
    if not sender_email or not password: return
    msg = MIMEText(f"OneSwifty Alert: You have spent ${current_spend:.2f}, which is 80% of your daily limit.")
    msg['Subject'] = "🚀 OneSwifty Budget Alert"
    msg['From'] = sender_email
    msg['To'] = USER_EMAIL
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, USER_EMAIL, msg.as_string())
    except Exception as e: print(f"Email failed: {e}")

def get_total_spend_today():
    log_file = "oneswifty_audit_log.csv"
    if not os.path.isfile(log_file): return 0.0
    try:
        df = pd.read_csv(log_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        today_df = df[df['Timestamp'].dt.date == datetime.now().date()]
        return today_df['Cost_USD'].astype(float).sum()
    except: return 0.0

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

# --- MAIN INTERFACE ---
st.title("🚀 OneSwifty: Universal Knowledge Engine[BETA]")

# --- INSTRUCTIONS ---
st.markdown("""
### 📖 How to use OneSwifty:
1. **Ingest Knowledge**: Use Step 1 to upload a PDF. AI auto-identifies the Title, Author, and Category.
2. **Verify Library**: Check Step 2 to ensure your knowledge base is populated.
3. **Search & Audit**: Use Step 3 to ask technical or financial questions.
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
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                total_pages = len(doc)
                first_page_sample = doc[0].get_text()[:1500]
                
                with st.spinner("AI analyzing document identity..."):
                    meta_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "You are a technical librarian. Extract the ACTUAL FORMAL TITLE,the FULL LIST of all authors (comma-separated), and a one-word CATEGORY. Return: Title | Author1, Author2, etc. | Category"},
                                  {"role": "user", "content": first_page_sample}]
                    )
                    metadata_raw = meta_response.choices[0].message.content
                try:
                    auto_title, auto_author, auto_category = metadata_raw.split("|")
                    auto_title, auto_author, auto_category = auto_title.strip(), auto_author.strip(), auto_category.strip()
                except ValueError:
                    auto_title, auto_author, auto_category = uploaded_file.name, "Unknown", "General"

                conn = get_connection()
                if conn:
                    with conn.cursor() as cur:
                        for i, page in enumerate(doc):
                            progress_bar.progress((i + 1) / total_pages)
                            status_text.text(f"Processing page {i+1} of {total_pages}...")
                            
                            text = page.get_text()
                            if text and len(text.strip()) > 20:
                                clean_text = text.replace("\x00", "")
                                chunk_size, overlap, start = 800, 80, 0
                                while start < len(clean_text):
                                    end = min(start + chunk_size, len(clean_text))
                                    chunk = clean_text[start:end].strip()
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
                    conn.close()

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

# --- FORMATTING FUNCTIONS ---

def clean_math_formatting(text):
    """
    Final standardized scrubber for OneSwifty.
    Fixes squashed text, double-rendering, and escape errors.
    """
    # 1. PREVENT DOUBLE-TEXT: Remove Zero-Width Spaces (the cause of fMG(z)fMG(z))
    text = text.replace("\u200b", "")
    
    # 2. RESCUE SQUASHED TEXT: If a whole paragraph is wrapped in $$, strip it.
    # LaTeX blocks ($$) remove spaces; sentences should be regular text.
    if text.count("$$") >= 2:
        text = re.sub(r"\$\$(.{150,})\$\$", r"\1", text, flags=re.DOTALL)

    # 3. FIX SPECIFIC SYMBOLS: Use double-backslashes (\\) to avoid 'bad escape' errors
    # This prevents the \d (delta) and \f (fraction) crashes.
    replacements = {
        r"fMG\(z\)\s*fMG\s*\(z\)": r"f_{MG}(z)",
        r"fMG\(a\)\s*fMG\s*\(a\)": r"f_{MG}(a)",
        r"δE\s*δE": r"\\delta_E",
        r"δmin": r"\\delta_{min}"
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    # 4. STANDARDIZE DELIMITERS: Convert AI brackets into $ wrappers
    text = text.replace(r"\[", "$$").replace(r"\]", "$$")
    text = text.replace(r"\(", "$").replace(r"\)", "$")
    
    # 5. WRAP NAKED VARIABLES: Ensures physics terms are formatted properly
    vars_to_fix = {
        "fMG": r"f_{MG}",
        "δE": r"\\delta_E",
        "z_in": r"z_{in}",
        "δmin": r"\\delta_{min}"
    }
    for raw, latex in vars_to_fix.items():
        # Using a lambda here is the only way to avoid 'bad escape' errors in Python
        pattern = f"(?<!\\$){re.escape(raw)}(?!\\$)"
        text = re.sub(pattern, lambda m, l=latex: f"${l}$", text)

    return text

def render_document_audit(text):
    """Renders the main audit container with cleaned LaTeX."""
    clean_text = clean_math_formatting(text)
    with st.container(border=True):
        st.markdown("### 📑 OneSwifty Document Audit")
        st.markdown(clean_text)
        st.caption("🔍 Precision Audit based on Source Documents")

def extract_key_findings(text):
    """Extracts bullets and cleans math for the insights expander."""
    bullet_pattern = r"(?:^|\n)\s*[\*•\-]\s*(.*)"
    findings = re.findall(bullet_pattern, text)
    
    if findings:
        unique_findings = list(dict.fromkeys([f.strip() for f in findings if f.strip()]))
        with st.expander("📝 OneSwifty: Key Insights & Highlights", expanded=True):
            for point in unique_findings[:5]: 
                cleaned_point = clean_math_formatting(point)
                # If the math is a big block, don't put a bullet in front of it
                if cleaned_point.startswith("$$"):
                    st.markdown(cleaned_point)
                else:
                    st.markdown(f"**•** {cleaned_point}")

def render_document_audit(text):
    # Pass it through the cleaner one last time to be safe
    text = clean_math_formatting(text)
    
    with st.container(border=True):
        st.markdown("### 📑 OneSwifty Document Audit")
        # Use st.write or st.markdown; both handle KaTeX well if delimiters are clean
        st.markdown(text)
        st.caption("🔍 Precision Audit based on Source Documents")

def extract_key_findings(text):
    """
    Scans the AI's response for any bulleted lists and cleans them 
    using the master math scrubber before rendering.
    """
    # 1. Capture lines starting with bullets
    bullet_pattern = r"(?:^|\n)\s*[\*•\-]\s*(.*)"
    findings = re.findall(bullet_pattern, text)
    
    if findings:
        # 2. Remove duplicates while preserving order
        unique_findings = list(dict.fromkeys([f.strip() for f in findings if f.strip()]))
        
        with st.expander("📝 OneSwifty: Key Insights & Highlights", expanded=True):
            for point in unique_findings[:5]: 
                # 3. Clean the math for each point
                cleaned_point = clean_math_formatting(point)
                
                # 4. UI FIX: If the cleaned point starts with $$ (block math),
                # we don't put a bullet in front of it to avoid rendering glitches.
                if cleaned_point.startswith("$$"):
                    st.markdown(cleaned_point)
                else:
                    st.markdown(f"**•** {cleaned_point}")




# --- STEP 3: SEARCH  ---
if is_over_budget:
    st.error(f"🛑 Daily Budget Reached (${DAILY_BUDGET_LIMIT}). Search is disabled.")
else:
    st.subheader("🔍 Step 3: Intelligent Search")
    query = st.chat_input("What would you like to ask about the uploaded files?")
    
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
                        ORDER BY similarity DESC LIMIT 10
                    """, (query_vec,))
                    results = cur.fetchall()
                    
                    if results:
                        context_text = "\n\n".join([f"TITLE: {res[2]} | AUTHORS: {res[4]} | PAGE: {res[5]} | CONTENT: {res[1]}" for res in results])
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system", 
                                    "content": """You are OneSwifty AI, a Universal Knowledge Engine and high-precision Auditor. 

MANDATORY CITATION RULE:
Every factual statement MUST be cited using the format: 'As seen on Page [X] in [Full Paper Title] by [Primary Author]...'

FORMATTING RULE:
Always include a brief bulleted list (using standard • bullets) summarizing the 3-4 most critical points of your answer, regardless of the topic (finance, literature, or science).

MATH & FORMULA RULE (STRICT):
- Every mathematical variable or Greek letter MUST be wrapped in single dollar signs (e.g., $\delta_{min}(z)$ or $f_{MG}(z)$).
- Every standalone equation or fraction MUST be on its own line and wrapped in double dollar signs (e.g., $$\delta_{min}(z) = \max(-1, ...)$$).
- NEVER use raw text for math like "delta_{min}". 
- Use standard LaTeX backslashes for Greek letters (e.g., \delta instead of delta).

FINANCIAL RULES (If Applicable):
When asked for a category total, look specifically for a line that contains the word 'Total'. 
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
                            st.markdown(f"**Confidence Score:** `{best_score*100:.2f}%`")
                            
                            if best_score < 0.50:
                                st.warning(f"⚠️ **Low Confidence Match ({best_score*100:.1f}%)**")
                            
                            # CLEAN THE ENTIRE ANSWER FIRST
                            
                            scrubbed_answer = clean_math_formatting(final_answer)
                            # Then pass the perfectly clean text to your UI functions
                            extract_key_findings(scrubbed_answer)
                            render_document_audit(scrubbed_answer)
                                
                            with st.expander("🔍 Document Audit Logs"):
                                st.write(f"**Primary Match Score:** {best_score*100:.2f}%")
                                for res in results:
                                    st.caption(f"**Doc:** {res[2]} | **Page:** {res[5]} | **Relevance:** {res[3]*100:.1f}%")
                    else:
                        st.error("No knowledge found.")

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
