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
st.title("🚀 OneSwifty: Universal Knowledge Engine[Testing]")

# --- INSTRUCTIONS ---
st.markdown("""
### 📖 How to use OneSwifty:
1. **Ingest Knowledge**: Use Step 1 to upload a PDF. AI auto-identifies the Title, Author, and Category.
2. **Verify Library**: Check Step 2 to ensure your knowledge base is populated.
3. **Search & Audit**: Use Step 3 to ask technical or financial questions with mandatory citations.
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

# --- STEP 1: INGESTION (RETORED PRECISION) ---
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
                        messages=[{"role": "system", "content": "Extract Title | Institutional Author | Category. Format: Title | Author | Category"},
                                  {"role": "user", "content": first_page_sample}]
                    )
                    try:
                        res_text = meta_response.choices[0].message.content
                        auto_title, auto_author, auto_category = res_text.split("|")
                    except:
                        auto_title, auto_author, auto_category = uploaded_file.name, "Unknown", "General"
                
                conn = get_connection()
                if conn:
                    with conn.cursor() as cur:
                        for i, page in enumerate(doc):
                            progress_bar.progress((i + 1) / total_pages)
                            status_text.text(f"Processing page {i+1} of {total_pages}...")
                            
                            text = page.get_text()
                            if text and len(text.strip()) > 20:
                                # Clean NULL bytes to prevent psycopg errors
                                clean_text = text.replace("\x00", "")
                                chunk_size, overlap, start = 800, 80, 0
                                while start < len(clean_text):
                                    end = min(start + chunk_size, len(clean_text))
                                    chunk = clean_text[start:end].strip()
                                    if len(chunk) > 20:
                                        vec = get_embedding(chunk)
                                        cur.execute("""INSERT INTO oneswifty_knowledge 
                                                    (category, content_text, metadata_source, author, title, page_number, embedding)
                                                    VALUES (%s, %s, %s, %s, %s, %s, %s)""", 
                                                    (auto_category.strip(), chunk, uploaded_file.name, auto_author.strip(), auto_title.strip(), i + 1, vec))
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

import re

import re

def render_scientific_audit(text):
    """
    Aggressively cleans Unicode artifacts and wraps math in LaTeX 
    for a high-precision Auditor interface.
    """
    # 1. HARD STRIP: Replace known problematic Unicode with pure LaTeX
    # This stops the 'μNL μNL' glitch at the source
    cleaning_map = {
        "μNL": r"$\mu_{NL}$",
        "μ": r"$\mu$",
        "δE": r"$\delta_E$",
        "δ": r"$\delta$",
        "α": r"$\alpha$",
        "ρ": r"$\rho$",
        "π": r"$\pi$"
    }
    
    for key, value in cleaning_map.items():
        text = text.replace(key, value)

    # 2. FIX PARENTHESES MATH: Catch ( \mu_NL ) and turn into $ \mu_NL $
    text = re.sub(r'\((?=\s?\\)(.*?)\)', r'$\1$', text)
    
    # 3. PREVENT TRIPLE-WRAPPING: If we ended up with $$...$$, fix it
    text = text.replace("$$$", "$").replace("$$", "$") # Ensure single $ for inline
    # But restore double $$ for standalone blocks if they were intended
    # (Optional: only if you expect large formulas)

    # 4. FINAL RENDER IN A STYLED BOX
    with st.container(border=True):
        st.markdown("### 🔬 OneSwifty Scientific Audit")
        st.markdown(text)
        
import re

def render_scientific_audit(text):
    """
    Cleans LaTeX formatting and renders inside a high-precision box.
    """
    # Fix academic parentheses (\mu) -> $\mu$
    text = re.sub(r'\((?=\s?\\)(.*?)\)', r'$\1$', text)
    
    # Fix plain text variables (delta_E) -> $\delta_E$
    physics_vars = ['delta', 'mu', 'alpha', 'rho', 'pi', 'beta', 'lambda', 'gamma']
    for var in physics_vars:
        text = re.sub(rf'\({var}(.*?)\)', rf'$\\{var}\1$', text)

    # Wrap standalone Unicode in LaTeX
    unicode_map = {'μ': r'\mu', 'δ': r'\delta', 'α': r'\alpha', 'ρ': r'\rho'}
    for char, latex in unicode_map.items():
        text = re.sub(rf'(?<!\$){char}(?!\$)', f'${latex}$', text)

    # Render in a professional Auditor Box
    with st.container(border=True):
        st.markdown("### 🔬 OneSwifty Scientific Audit")
        st.markdown(text)
import re
import streamlit as st

def extract_key_findings(text):
    """
    Scans the AI response for specific scientific 'triggers' 
    and returns a clean bulleted list of the most critical points.
    """
    # Define our 'OneSwifty' high-value triggers
    triggers = [
        r".*?stronger effective gravity.*?\.",
        r".*?μNL > 1.*?\.",
        r".*?sub-?percent level.*?\.",
        r".*?deeper voids.*?\.",
        r".*?initial conditions.*?\.",
        r".*?linear growth.*?\."
    ]
    
    findings = []
    for pattern in triggers:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Clean up any weird LaTeX formatting for the summary bullet
            clean_bullet = match.group(0).strip().replace("(", "").replace(")", "")
            findings.append(clean_bullet)
    
    # Render the summary if we found hits
    if findings:
        with st.expander("📝 Quick Audit Summary", expanded=True):
            for point in findings[:4]: # Limit to top 4 for cleanliness
                st.markdown(f"**•** {point}")

# --- STEP 3: SEARCH (FORCE LaTeX & SOURCE MAP) ---
if is_over_budget:
    st.error(f"🛑 Daily Budget Reached (${DAILY_BUDGET_LIMIT}). Search is disabled.")
else:
    st.subheader("🔍 Step 3: Intelligent Search")
    query = st.chat_input("Ask about MG vs GR hierarchy...")
    
    if query:
        st.chat_message("user").write(query)
        with st.spinner("Analyzing gravitational couplings..."):
            query_vec = get_embedding(query)
            conn = get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("""SELECT content_text, title, 1 - (embedding <=> %s::vector) AS sim, page_number 
                                FROM oneswifty_knowledge ORDER BY sim DESC LIMIT 5""", (query_vec,))
                    results = cur.fetchall()
                    
                    if results:
                        # Fixed: consolidated context string
                        context = "\n\n".join([f"DOC: {r[1]} | PAGE: {r[3]} | CONTENT: {r[0]}" for r in results])
                        
                        resp = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system", 
                                    "content": r"""You are OneSwifty AI, a high-precision Scientific Auditor. 
            
                                    CRITICAL FORMATTING RULES:
                                    1. STANDALONE MATH: Use double dollar signs for equations: $$ [Formula] $$
                                    2. LATEX ONLY: Never use Unicode Greek characters (μ, δ). Always use $\mu$, $\delta$.
                                    3. NO BRACKETS: Never use \[ \] or \( \).
                                    4. VAINSHTEIN: Correct form is $$(R_V/R)^3 = \frac{32\pi}{3} G \beta^2 \lambda^2 \bar{\rho}_m \delta_v$$
                                    STRICT FORMATTING RULES:
                                    1. INLINE MATH: You MUST use single dollar signs for all variables. 
                                       - WRONG: (\mu_{NL}) or (delta_E)
                                       - RIGHT: $\mu_{NL}$ or $\delta_E$
                                    2. STANDALONE MATH: Use double dollar signs: $$ [Formula] $$
                                    3. NO PARENTHESES FOR MATH: Never wrap LaTeX in standard brackets or parentheses.
                                    4. SYMBOL ACCURACY: Ensure \mu_{NL} is used for non-linear coupling and \delta_E for overdensity.
                                    """
                                },
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                            ]
                        )
                        
                        answer = resp.choices[0].message.content
                        
                        # --- TECHNICAL SUMMARY & SOURCE MAP ---
                        if any(x in answer for x in ["Modified Gravity", "MG"]):
                            st.info("### 🔬 Technical Summary: MG Hierarchy & Coupling")
                            st.latex(r"max_{0\le z\le z_{in}}f_{MG}(z)>1")
                            st.markdown("""
                            **Hierarchy of Effects:**
                            * **Parameter Impact:** Increasing $\\alpha_{B0}$ or $m$ strengthens effective gravity.
                            * **Evolution Efficiency:** MG progresses faster, requiring smaller ICs for fixed $\\delta_E$.
                            * **Void Scaling:** MG deviations are amplified in deeper voids.
                            """)
                            
                            with st.expander("📍 Source Map: Cross-Document Evidence", expanded=False):
                                source_data = pd.DataFrame(results, columns=["Text", "Document", "Similarity", "Page"])
                                # Optional: Convert similarity to percentage for better UX
                                source_data["Similarity"] = source_data["Similarity"].apply(lambda x: f"{x*100:.1f}%")
                                st.table(source_data[["Document", "Page", "Similarity"]])

                        with st.chat_message("assistant"):
                        if answer:
                        # Show the quick summary first
                        extract_key_findings(answer)
        
                        # Then show the full detailed audit in the border box
                        render_scientific_audit(answer)                        
                        
                        log_query(query, answer, 0.9, resp.usage.prompt_tokens, resp.usage.completion_tokens)
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
