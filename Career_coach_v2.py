import streamlit as st
import pandas as pd
from typing import List
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import webbrowser

# ----------------------------
# --- INSERT YOUR GPT-4 MINI KEY HERE ---
# ----------------------------
openai.api_key = "YOUR_GPT4_MINI_KEY"  # <<< Replace this with your GPT-4-mini key

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Agentic AI Career Coach",
    page_icon="🚀",
    layout="wide",
)

# ----------------------------
# Load Data
# ----------------------------
resume_df = pd.read_csv("resume_data.csv")       # <<< Replace with your CSV path
mentors_df = pd.read_excel("mentors_list.xlsx")  # <<< Replace with your Excel path

# ----------------------------
# Helper Functions
# ----------------------------
def extract_skills_from_resume(df: pd.DataFrame) -> dict:
    role_skills = {}
    for _, row in df.iterrows():
        role = str(row['Role']).strip()
        skills = [s.strip().title() for s in str(row['Skills']).split(',') if s.strip()]
        role_skills[role] = skills
    return role_skills

role_skills_map = extract_skills_from_resume(resume_df)

def compute_role_embeddings(role_skills_map: dict) -> dict:
    embeddings = {}
    for role, skills in role_skills_map.items():
        text = role + " " + ", ".join(skills)
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings[role] = np.array(response['data'][0]['embedding'])
    return embeddings

role_embeddings = compute_role_embeddings(role_skills_map)

def recommend_role_with_embeddings(user_skills: List[str], role_embeddings: dict, threshold=0.7):
    user_text = ", ".join(user_skills)
    response = openai.Embedding.create(
        model="text-embedding-3-small",
        input=user_text
    )
    user_emb = np.array(response['data'][0]['embedding'])
    
    best_role = None
    best_score = -1
    for role, emb in role_embeddings.items():
        score = cosine_similarity([user_emb], [emb])[0][0]
        if score > best_score:
            best_score = score
            best_role = role
    
    if best_score < threshold:
        return None, best_score
    return best_role, best_score

def compute_skill_gap(user_skills: List[str], target_role: str, role_skills_map: dict) -> List[str]:
    target_skills = role_skills_map.get(target_role, [])
    user_skills_set = set([s.strip().title() for s in user_skills if s.strip()])
    missing = [s for s in target_skills if s not in user_skills_set]
    return missing

def suggest_learning_plan(missing_skills: List[str], target_role: str, mentors_df: pd.DataFrame) -> List[str]:
    prompt = f"""
    I am designing a personalized learning roadmap.
    Target Role: {target_role}
    Missing Skills: {', '.join(missing_skills)}
    Mentors available: {mentors_df['Mentor'].tolist() if not mentors_df.empty else 'No mentors yet'}
    Suggest a 3-4 step roadmap with courses/projects and recommended mentors if available.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    plan_text = response['choices'][0]['message']['content']
    plan_steps = [line.strip("- ").strip() for line in plan_text.split("\n") if line.strip()]
    return plan_steps

def fallback_web_search(user_skills: List[str]):
    query = "top roles for skills " + ", ".join(user_skills)
    url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    webbrowser.open_new_tab(url)
    return f"No close match found in CSV. Opened web search: {url}"

def generate_pdf_report(report_data: dict, filename="career_plan.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Personalized Career Growth Plan", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Name: {report_data['name']}", ln=True)
    pdf.cell(0, 8, f"Current Role: {report_data['current_role']} ({report_data['years_experience']} yrs)", ln=True)
    pdf.cell(0, 8, f"Desired Role: {report_data['desired_role']}", ln=True)
    pdf.cell(0, 8, f"Recommended Role: {report_data['recommended_role']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Skill Gaps:", ln=True)
    pdf.set_font("Arial", '', 12)
    if report_data['skill_gap']:
        for skill in report_data['skill_gap']:
            pdf.cell(0, 8, f"- {skill}", ln=True)
    else:
        pdf.cell(0, 8, "None — great fit!", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Suggested Learning Roadmap:", ln=True)
    pdf.set_font("Arial", '', 12)
    for idx, step in enumerate(report_data['learning_plan'], 1):
        pdf.multi_cell(0, 8, f"{idx}. {step}")
    
    pdf.output(filename)
    return filename

# ----------------------------
# Streamlit Sidebar Input
# ----------------------------
st.sidebar.title("Your Inputs")
name = st.sidebar.text_input("Your Name", "Alex")
current_role = st.sidebar.text_input("Current Role", "Analyst")
years_exp = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)
skills_text = st.sidebar.text_area("Top Skills (comma-separated)", "SQL, Excel, Python")
desired_role = st.sidebar.text_input("Desired Role (next 1-2 yrs)", "Data Scientist")

submitted = st.sidebar.button("Generate Career Plan")

# ----------------------------
# Main Output
# ----------------------------
if submitted:
    user_skills_list = [s.strip() for s in skills_text.split(",") if s.strip()]
    
    # Step 1: Embedding-based recommendation
    recommended_role, score = recommend_role_with_embeddings(user_skills_list, role_embeddings)
    
    if recommended_role:
        st.info(f"Recommended Role (semantic match): {recommended_role} (similarity: {score:.2f})")
    else:
        fallback_msg = fallback_web_search(user_skills_list)
        st.warning(fallback_msg)
        recommended_role = "No CSV match, see web search"
    
    # Step 2: Compute skill gap
    skill_gap = compute_skill_gap(user_skills_list, desired_role, role_skills_map) if recommended_role != "No CSV match, see web search" else []
    
    # Step 3: GPT-4-mini learning plan
    learning_plan = suggest_learning_plan(skill_gap, desired_role, mentors_df) if skill_gap else []
    
    # Display Results
    st.header(f"Career Growth Plan for {name}")
    st.subheader("🎯 Desired Role")
    st.success(desired_role)
    
    st.subheader("🧭 Recommended Role")
    st.info(recommended_role)
    
    st.subheader("🧩 Skill Gaps to Close")
    if skill_gap:
        for skill in skill_gap:
            st.markdown(f"- {skill}")
    else:
        st.success("No major skill gaps detected!")
    
    st.subheader("🗺️ Suggested Learning Roadmap")
    if learning_plan:
        for idx, step in enumerate(learning_plan, 1):
            st.markdown(f"{idx}. {step}")
    else:
        st.warning("Learning plan could not be generated.")
    
    # Step 4: Generate PDF
    report_data = {
        "name": name,
        "current_role": current_role,
        "years_experience": years_exp,
        "desired_role": desired_role,
        "recommended_role": recommended_role,
        "user_skills": user_skills_list,
        "skill_gap": skill_gap,
        "learning_plan": learning_plan
    }
    
    pdf_file = generate_pdf_report(report_data)
    st.download_button(
        label="⬇️ Download PDF",
        data=open(pdf_file, "rb").read(),
        file_name="career_plan.pdf",
        mime="application/pdf"
    )
