import streamlit as st
from typing import List, Dict, Tuple
import json

# ----------------------------
# Page config & basic styling
# ----------------------------
st.set_page_config(
    page_title="Agentic AI Career Coach — MVP",
    page_icon="🚀",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/***** Global *****/
:root {
  --card-bg: #0b1220; /* deep navy */
  --gradient: linear-gradient(135deg, #5b9dff 0%, #8a7dff 50%, #3fe0c5 100%);
}

/* App background */
section.main > div {background: radial-gradient(1200px 600px at 0% 0%, rgba(91,157,255,.08), transparent),
                                   radial-gradient(1200px 600px at 100% 0%, rgba(63,224,197,.05), transparent);
}

/* Headline gradient text */
.gradient-text {background: var(--gradient); -webkit-background-clip: text; background-clip: text; color: transparent;}

/* Card look */
.card {background: rgba(11,18,32,.7); border: 1px solid rgba(255,255,255,.06); border-radius: 18px; padding: 18px 20px; box-shadow: 0 10px 30px rgba(0,0,0,.25);} 
.card h3 {margin-top: 0.2rem;}

/* Tag/Pill */
.pill {display:inline-block; padding:6px 10px; border-radius:999px; margin:4px 6px 0 0; font-size:0.9rem; border:1px solid rgba(255,255,255,.12);}
.pill.high {background: rgba(255, 99, 71, .15);} /* tomato */
.pill.med {background: rgba(255, 206, 86, .15);} /* amber */
.pill.low {background: rgba(75, 192, 192, .15);}  /* teal */

/* Roadmap step */
.step {display:flex; align-items:center; gap:10px;}
.step .dot {width:12px; height:12px; border-radius:50%; background: #8a7dff;}
.step .arrow {height:2px; flex:1; background: linear-gradient(90deg, rgba(138,125,255,.6), rgba(63,224,197,.6));}

.small-muted {opacity:.75; font-size:0.92rem}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Minimal role/skill database (MVP placeholder)
# ----------------------------
# Each role has required skills and suggested learning resources.
ROLE_DB: Dict[str, Dict[str, List[str]]] = {
    "Data Analyst": {
        "skills": ["SQL", "Excel", "Data Visualization", "Tableau", "Python"],
        "resources": [
            "SQL Basics — freeCodeCamp",
            "Excel for Data Analysis — Microsoft Learn",
            "Data Visualization — Storytelling with Data",
            "Tableau A-Z — Coursera",
            "Python for Data Analysis — Kaggle"
        ],
    },
    "Data Scientist": {
        "skills": ["Python", "Statistics", "Machine Learning", "Pandas", "SQL", "Model Deployment"],
        "resources": [
            "Python + Pandas — Kaggle",
            "Practical Statistics — Khan Academy",
            "Machine Learning — Andrew Ng (Coursera)",
            "Feature Engineering — fast.ai",
            "Intro to MLOps — Google"
        ],
    },
    "Product Manager": {
        "skills": ["Product Strategy", "User Research", "Roadmapping", "Analytics", "Stakeholder Management"],
        "resources": [
            "Product Management 101 — Reforge",
            "Lean UX — Jeff Gothelf",
            "Roadmaps — Gibson Biddle talks",
            "Product Analytics — Mode SQL tutorials",
            "Influence without Authority — Book"
        ],
    },
    "Frontend Engineer": {
        "skills": ["HTML", "CSS", "JavaScript", "React", "Testing", "Performance"],
        "resources": [
            "Modern JS — MDN",
            "CSS Layout — Josh Comeau",
            "React Official Docs",
            "Testing React — Testing Library",
            "Web Performance Fundamentals — web.dev"
        ],
    },
}

# Suggested intermediate steps for common transitions (very small heuristic map)
ROLE_TRANSITIONS: Dict[Tuple[str, str], List[str]] = {
    ("Analyst", "Data Scientist"): ["Data Analyst", "ML Associate"],
    ("Frontend Engineer", "Product Manager"): ["Tech PM", "Associate PM"],
}

# Priority weights for typical skill importance per role (optional, lightweight heuristic)
PRIORITY_WEIGHTS: Dict[str, Dict[str, int]] = {
    "Data Scientist": {"Python": 3, "Machine Learning": 3, "Statistics": 3, "Pandas": 2, "SQL": 2, "Model Deployment": 2},
    "Data Analyst": {"SQL": 3, "Tableau": 2, "Excel": 2, "Data Visualization": 2, "Python": 1},
    "Product Manager": {"Product Strategy": 3, "User Research": 2, "Roadmapping": 2, "Analytics": 2, "Stakeholder Management": 2},
    "Frontend Engineer": {"JavaScript": 3, "React": 3, "HTML": 2, "CSS": 2, "Testing": 2, "Performance": 2},
}

# ----------------------------
# Helper functions
# ----------------------------

def normalize_role(role: str) -> str:
    return (role or "").strip().title()


def compute_skill_gap(user_skills: List[str], target_role: str) -> List[Tuple[str, str, int]]:
    """Return list of (skill, priority_label, weight) for missing skills, sorted by priority."""
    t = normalize_role(target_role)
    required = ROLE_DB.get(t, {}).get("skills", [])
    weights = PRIORITY_WEIGHTS.get(t, {})

    missing = []
    have = {s.strip().title() for s in user_skills if s.strip()}
    for sk in required:
        label = sk.title()
        if label not in have:
            w = weights.get(label, 1)
            priority = "High" if w >= 3 else ("Medium" if w == 2 else "Low")
            missing.append((label, priority, w))
    # Sort by weight desc, then alphabetical
    missing.sort(key=lambda x: (-x[2], x[0]))
    return missing


def propose_roadmap(current_role: str, target_role: str) -> List[str]:
    c = (current_role or "").strip()
    t = (target_role or "").strip()
    steps = []
    # Exact map first
    steps = ROLE_TRANSITIONS.get((c, t), [])
    # Fallback: if target exists, propose one intermediate if available
    if not steps and normalize_role(target_role) in ROLE_DB:
        if normalize_role(target_role) == "Data Scientist" and "Data Analyst" in ROLE_DB:
            steps = ["Data Analyst"]
    return [c] + steps + [t]


def learning_recs(target_role: str, missing_skills: List[Tuple[str, str, int]]) -> List[str]:
    base = ROLE_DB.get(normalize_role(target_role), {}).get("resources", [])
    # Show 3–5 items; emphasize first two missing skills in text
    picks = base[:5]
    return picks


def generate_markdown_report(name: str, current_role: str, years: float, target_role: str, roadmap: List[str], gaps: List[Tuple[str,str,int]], recs: List[str]) -> str:
    gap_lines = [f"- **{s}** — _{p} priority_" for s,p,_ in gaps]
    step_str = " → ".join([f"**{s}**" for s in roadmap if s])
    rec_lines = [f"- {r}" for r in recs]
    md = f"""
# Personalized Career Growth Plan

**Name:** {name or '—'}  
**Current Role:** {current_role or '—'} ({years if years is not None else '—'} yrs)  
**Target Role:** {target_role or '—'}

---

## Career Path
{step_str}

## Skill Gaps (focus next)
{chr(10).join(gap_lines) if gap_lines else '- None — great fit!'}

## Learning Roadmap (3–5 picks)
{chr(10).join(rec_lines) if rec_lines else '- Coming soon'}
"""
    return md

# ----------------------------
# Sidebar — Input Form
# ----------------------------
with st.sidebar:
    st.markdown("<h2 class='gradient-text'>Your Inputs</h2>", unsafe_allow_html=True)
    name = st.text_input("Your name", placeholder="Alex")
    col_a, col_b = st.columns(2)
    with col_a:
        current_role = st.text_input("Current role", placeholder="Analyst / Frontend Engineer / ...")
    with col_b:
        years_exp = st.number_input("Years of experience", min_value=0.0, max_value=50.0, value=2.0, step=0.5)

    skills_text = st.text_area("Top skills (comma-separated)", placeholder="SQL, Excel, Python")
    desired_role = st.text_input("Desired role (next 1–2 yrs)", placeholder="Data Scientist")

    location = st.text_input("(Optional) Location", placeholder="Berlin, DE")

    submitted = st.button("Generate Plan", type="primary", use_container_width=True)

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <div style='margin: 10px 0 1.2rem 0;'>
        <h1 class='gradient-text' style='margin-bottom:0;'>Hi {}</h1>
        <div class='small-muted'>Your personalized career growth plan</div>
    </div>
    """.format(name or "there"),
    unsafe_allow_html=True,
)

# ----------------------------
# If not submitted, show a friendly prompt
# ----------------------------
if not submitted:
    st.info("Fill the form on the left and click **Generate Plan** to see your role recommendation, skill gaps, and a learning roadmap.")
    st.stop()

# ----------------------------
# Compute plan
# ----------------------------
skills_list = [s.strip() for s in (skills_text.split(",") if skills_text else []) if s.strip()]

roadmap_steps = propose_roadmap(current_role, desired_role)
missing = compute_skill_gap(skills_list, desired_role)
recs = learning_recs(desired_role, missing)

# ----------------------------
# Layout — 3 cards in responsive grid
# ----------------------------
col1, col2 = st.columns([1.1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("""
    <h3>🎯 Your Next Career Role</h3>
    <div class='small-muted'>Based on your current role and target.</div>
    """, unsafe_allow_html=True)

    # Roadmap visualization
    st.write("")
    for i, step in enumerate(roadmap_steps):
        if not step:
            continue
        cols = st.columns([0.05, 0.95])
        with cols[0]:
            st.markdown("<div class='dot'></div>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<div class='step'><strong>{step}</strong></div>", unsafe_allow_html=True)
        if i < len(roadmap_steps) - 1:
            st.markdown("<div class='arrow'></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3>🧩 Skill Gaps to Close</h3>", unsafe_allow_html=True)

    if missing:
        for skill, priority, weight in missing[:5]:
            level = "high" if priority == "High" else ("med" if priority == "Medium" else "low")
            st.markdown(f"<span class='pill {level}'>{skill} — <em>{priority}</em></span>", unsafe_allow_html=True)
    else:
        st.success("No major gaps detected for this target — you're in great shape! ✅")

    st.markdown("</div>", unsafe_allow_html=True)

# Learning roadmap full-width
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h3>🗺️ 3–5 Learning Recommendations</h3>", unsafe_allow_html=True)
if recs:
    for idx, r in enumerate(recs, start=1):
        st.markdown(f"**{idx}.** {r}")
else:
    st.warning("No resources available for this role yet. Add items to ROLE_DB to expand.")

st.markdown("<div class='small-muted'>Tip: Prioritize High → Medium → Low skills. Consistency beats intensity.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Export — Markdown & JSON
# ----------------------------
report_md = generate_markdown_report(name, current_role, years_exp, desired_role, roadmap_steps, missing, recs)

colx, coly, colz = st.columns([1,1,1])
with colx:
    st.download_button(
        label="⬇️ Download Plan (Markdown)",
        data=report_md.encode("utf-8"),
        file_name="career_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )
with coly:
    payload = {
        "name": name,
        "current_role": current_role,
        "years_experience": years_exp,
        "desired_role": desired_role,
        "location": location,
        "skills_input": skills_list,
        "roadmap_steps": roadmap_steps,
        "skill_gaps": [{"skill": s, "priority": p, "weight": w} for s,p,w in missing],
        "learning_recommendations": recs,
    }
    st.download_button(
        label="⬇️ Download JSON",
        data=json.dumps(payload, indent=2).encode("utf-8"),
        file_name="career_plan.json",
        mime="application/json",
        use_container_width=True,
    )
with colz:
    st.caption("ℹ️ Extend the ROLE_DB to add more roles/skills/resources.")

# ----------------------------
# Empty state nudges / next steps
# ----------------------------
st.markdown("""
<div class='small-muted'>
<strong>How to extend this MVP:</strong> Add roles to <code>ROLE_DB</code>, tune <code>PRIORITY_WEIGHTS</code>, and enhance <code>ROLE_TRANSITIONS</code>. For production, swap placeholder data with O*NET/ESCO and generate resources dynamically.
</div>
""", unsafe_allow_html=True)
