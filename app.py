import streamlit as st
import spacy
from spacy.matcher import Matcher, PhraseMatcher
import re
import pdfplumber
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- Load NLP Model ---
# Hum seedha load kar rahe hain kyunki yeh requirements.txt se install ho jayega
MODEL_NAME = "en_core_web_md"
try:
    nlp = spacy.load(MODEL_NAME)
except OSError:
    st.error(f"Model '{MODEL_NAME}' load nahi ho pa raha. Check karein ki woh requirements.txt mein sahi se add hai.")
    st.stop()


# --- Configuration (Skills & Keywords) ---
SKILLS_DB = [
    'Python', 'Java', 'C++', 'JavaScript', 'React', 'Node.js', 'Angular',
    'Streamlit', 'Flask', 'Django', 'Pandas', 'NumPy', 'SciPy', 'Matplotlib',
    'scikit-learn', 'TensorFlow', 'PyTorch', 'Keras', 'NLP', 'spaCy', 'NLTK',
    'Machine Learning', 'Deep Learning', 'Data Analysis', 'Data Science',
    'SQL', 'MySQL', 'PostgreSQL', 'MongoDB', 'Firebase', 'AWS', 'Azure', 'GCP',
    'Docker', 'Kubernetes', 'Git', 'GitHub', 'CI/CD', 'Agile', 'Scrum', 'JIRA'
]

MUST_HAVE_KEYWORDS = ['required', 'must-have', 'essential', 'critical', 'mandatory']
GOOD_TO_HAVE_KEYWORDS = ['preferred', 'plus', 'bonus', 'nice-to-have', 'advantageous']
ACTION_VERBS = ['led', 'managed', 'developed', 'created', 'implemented', 'achieved', 'optimized', 'automated', 'designed', 'built']


# --- 1. Text Extraction Functions ---

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_docx(file)
    return ""

# --- 2. "Pro" Parsing Functions ---

def parse_jd_requirements(jd_text):
    jd_doc = nlp(jd_text.lower())
    must_have_skills = set()
    good_to_have_skills = set()
    
    for sent in jd_doc.sents:
        sent_text = sent.text
        if any(keyword in sent_text for keyword in MUST_HAVE_KEYWORDS):
            for skill in SKILLS_DB:
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', sent_text):
                    must_have_skills.add(skill)
        if any(keyword in sent_text for keyword in GOOD_TO_HAVE_KEYWORDS):
            for skill in SKILLS_DB:
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', sent_text):
                    good_to_have_skills.add(skill)

    if not must_have_skills and not good_to_have_skills:
        for skill in SKILLS_DB:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', jd_text.lower()):
                must_have_skills.add(skill)
                
    return list(must_have_skills), list(good_to_have_skills)

def extract_skills_from_resume(resume_text):
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(skill) for skill in SKILLS_DB]
    matcher.add("SKILLS", patterns)
    doc = nlp(resume_text)
    matches = matcher(doc)
    found_skills = set()
    for match_id, start, end in matches:
        found_skills.add(doc[start:end].text.title())
    return list(found_skills)

def extract_years_of_experience(resume_text):
    matches = re.findall(r'(\d+)\s*(?:\+|years|yrs)\s*experience', resume_text, re.IGNORECASE)
    if matches:
        return max([int(m) for m in matches])
    return 0

def check_action_verbs(resume_text):
    doc = nlp(resume_text)
    action_verb_count = 0
    total_bullet_points = 0
    for line in resume_text.split('\n'):
        line = line.strip()
        if line.startswith(('*', '-', '•')):
            total_bullet_points += 1
            first_word_match = re.match(r'^\W*(\w+)', line)
            if first_word_match:
                first_word = first_word_match.group(1).lower()
                if first_word in ACTION_VERBS:
                    action_verb_count += 1
    if total_bullet_points == 0:
        return 0
    return int((action_verb_count / total_bullet_points) * 100)


# --- 3. "Pro" Scoring Engine ---

def calculate_pro_score(resume_text, jd_text, resume_skills, jd_must_have, jd_good_to_have, resume_yoe):
    scores = {'must_have': 0, 'good_to_have': 0, 'contextual': 0, 'experience': 0}
    
    if jd_must_have:
        matched_must_have = set(resume_skills) & set(jd_must_have)
        scores['must_have'] = (len(matched_must_have) / len(jd_must_have)) * 100
    else:
        scores['must_have'] = 100

    if jd_good_to_have:
        matched_good_to_have = set(resume_skills) & set(jd_good_to_have)
        scores['good_to_have'] = (len(matched_good_to_have) / len(jd_good_to_have)) * 100
    else:
        scores['good_to_have'] = 100

    doc_resume = nlp(resume_text)
    doc_jd = nlp(jd_text)
    scores['contextual'] = doc_resume.similarity(doc_jd) * 100

    jd_yoe_match = re.search(r'(\d+)\s*(?:\+|years|yrs)', jd_text, re.IGNORECASE)
    if jd_yoe_match:
        jd_yoe = int(jd_yoe_match.group(1))
        if resume_yoe >= jd_yoe:
            scores['experience'] = 100
        elif jd_yoe > 0:
            scores['experience'] = (resume_yoe / jd_yoe) * 100
    else:
        scores['experience'] = 100

    final_score = (scores['must_have'] * 0.5) + (scores['good_to_have'] * 0.2) + \
                  (scores['contextual'] * 0.2) + (scores['experience'] * 0.1)
                  
    return int(final_score), scores

# --- 4. Feedback Generator ---

def generate_pro_feedback(scores, resume_skills, jd_must_have, jd_good_to_have, action_verb_score):
    feedback = []
    missing_must_have = set(jd_must_have) - set(resume_skills)
    if missing_must_have:
        feedback.append(f"**Critical Missing Skills:** Aapko *must-have* skills ({', '.join(missing_must_have)}) ko apne resume mein prominently add karna chahiye. Iske bina selection mushkil hai.")
    else:
        feedback.append("👍 **Great!** Aapke paas saari *must-have* skills hain.")
        
    missing_good_to_have = set(jd_good_to_have) - set(resume_skills)
    if missing_good_to_have:
        feedback.append(f"**Pro Tip (Bonus Points):** Agar aap in *good-to-have* skills ({', '.join(missing_good_to_have)}) mein se kuch add kar sakein, toh aapke chances badh jayenge.")

    if scores['contextual'] < 50:
        feedback.append(f"**Language Mismatch:** Aapka resume (Score: {scores['contextual']:.0f}%) JD ki language se kam match kar raha hai. *Suggestion: JD mein use kiye gaye keywords (jaise 'optimization', 'leadership') ko apne resume mein bhi istemal karein.*")

    if action_verb_score < 60:
        feedback.append(f"**Weak Lingo:** Aapke {action_verb_score}% bullet points hi strong action verbs (Led, Developed) se start hote hain. *Suggestion: 'Responsible for' ki jagah 'Managed', 'Developed', 'Achieved' jaise words ka prayog karein.*")
    else:
        feedback.append("💪 **Strong Lingo:** Aapke resume ke bullet points strong action verbs use karte hain. Accha kaam!")
        
    return feedback


# --- 5. Streamlit UI (Main Application) ---

st.set_page_config(layout="wide")
st.title("Project Pro: The 'Recruiter's Eye' Resume Analyzer 👁️‍🗨️")
st.markdown("Yeh ek advanced analyzer hai jo sirf keywords nahi, balki context, experience, aur JD requirements ko deeply analyze karta hai.")

# --- DEMO BUTTON ---
SAMPLE_JD = """
**Job Title: Senior Python Developer (AI/ML)**
**Role:**
We are seeking a highly motivated Senior Python Developer to join our innovation team. The ideal candidate must-have 5+ years of experience in developing and deploying machine learning models.
**Key Responsibilities:**
- Design, build, and maintain efficient, reusable, and reliable Python code.
- Implement machine learning models using scikit-learn and TensorFlow.
- Develop backend services using Flask or Django.
- Manage data pipelines and databases (SQL, MongoDB).
- Work with cloud platforms, AWS is required.
**Qualifications:**
- **Required:** Python, Machine Learning, scikit-learn, SQL.
- **Preferred (Bonus):** Docker, AWS, NLP, Streamlit.
"""

SAMPLE_RESUME = """
**John Doe**
AI & Machine Learning Engineer | 5+ years experience
**Summary:**
Results-driven developer with 5 years of experience in building scalable AI solutions. Led multiple projects from concept to deployment.
**Experience:**
**Sr. AI Engineer (2020 - Present)**
- Developed a customer churn prediction model using Python, scikit-learn, and TensorFlow, achieving 92% accuracy.
- Built a REST API using Flask to serve model predictions.
- Managed a large PostgreSQL database for training data.
- Optimized data processing tasks using Pandas and NumPy.
**Skills:**
- **Languages:** Python, SQL
- **Frameworks:** scikit-learn, TensorFlow, Pandas, Flask
- **Tools:** Git, Docker, JIRA, Agile
"""

if 'jd_text_area' not in st.session_state:
    st.session_state.jd_text_area = ""
if 'demo_resume_text' not in st.session_state:
    st.session_state.demo_resume_text = ""

if st.button("Recruiter Busy? Click for a Quick Demo! 💡"):
    st.session_state.jd_text_area = SAMPLE_JD
    st.session_state.demo_resume_text = SAMPLE_RESUME
    st.rerun()

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("1. Apna Resume Upload Karein")
    resume_file = st.file_uploader("Sirf PDF ya DOCX file", type=['pdf', 'docx'], key="resume_file")
    
    if st.session_state.demo_resume_text and not resume_file:
        st.info("Demo Resume Loaded. Press 'Analyze' below.")
        st.text_area("Demo Resume Preview", st.session_state.demo_resume_text, height=200, disabled=True)

with col2:
    st.header("2. Job Description (JD) Paste Karein")
    jd_text = st.text_area("Yahan job description daalein...", 
                           value=st.session_state.jd_text_area, 
                           height=310, 
                           key="jd_text_area")

# --- Analysis Button ---
if st.button("Analyze My Hireability 🚀", type="primary"):
    
    # PEHLE DEMO CHECK KARO
    if st.session_state.demo_resume_text and jd_text:
        st.info("Analyzing Demo Resume...")
        resume_text = st.session_state.demo_resume_text
    
    # AGAR DEMO NAHI HAI, TAB FILE UPLOAD CHECK KARO
    elif resume_file is not None:
        resume_text = extract_text(resume_file)
        if not jd_text:
            st.error("Please job description paste karein.")
            st.stop()
            
    # AGAR KUCH BHI NAHI HAI
    else:
        st.error("Please resume file upload karein aur JD paste karein (ya demo button use karein).")
        st.stop()
        
    with st.spinner("Ek Recruiter ki tarah aapka resume padh raha hoon... 🧐"):
        jd_must_have, jd_good_to_have = parse_jd_requirements(jd_text)
        resume_skills = extract_skills_from_resume(resume_text)
        resume_yoe = extract_years_of_experience(resume_text)
        action_verb_score = check_action_verbs(resume_text)

        final_score, scores_breakdown = calculate_pro_score(
            resume_text, jd_text, resume_skills, 
            jd_must_have, jd_good_to_have, resume_yoe
        )
        
        feedback_points = generate_pro_feedback(
            scores_breakdown, resume_skills,
            jd_must_have, jd_good_to_have, action_verb_score
        )
        
    st.success("Analysis Complete!")
    
    st.header(f"Final Hireability Score: {final_score}%")
    st.progress(final_score)
    
    if final_score >= 80:
        st.markdown("### 🏆 Zabardast! Aap interview ke liye strong candidate hain.")
    elif final_score >= 60:
        st.markdown("### 👍 Accha hai, par thode improvements se top candidate ban sakte hain.")
    else:
        st.markdown("### 😐 Kaam karne ki zaroorat hai. Niche diye gaye feedback par focus karein.")
        
    st.divider()

    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.subheader("📊 Score Breakdown")
        df_scores = pd.DataFrame({
            'Category': ['Must-Have Skills', 'Good-to-Have Skills', 'Contextual Match', 'Experience Match'],
            'Weight': ['50%', '20%', '20%', '10%'],
            'Your Score': [
                f"{scores_breakdown['must_have']:.0f}%",
                f"{scores_breakdown['good_to_have']:.0f}%",
                f"{scores_breakdown['contextual']:.0f}%",
                f"{scores_breakdown['experience']:.0f}%"
            ]
        })
        st.table(df_scores)
        
        st.subheader("🛠️ Action Verb Analysis")
        st.markdown(f"Aapke **{action_verb_score}%** experience points strong verbs se shuru hote hain.")
        st.progress(action_verb_score)

    with res_col2:
        st.subheader("💡 Actionable Feedback")
        for feedback in feedback_points:
            st.markdown(f"- {feedback}")
    
    st.divider()
    
    with st.expander("Skill Match Details Dekhein"):
        skill_col1, skill_col2, skill_col3 = st.columns(3)
        
        all_jd_skills = set(jd_must_have) | set(jd_good_to_have)
        matched_skills = set(resume_skills) & all_jd_skills
        
        skill_col1.metric("Total Skills in JD", len(all_jd_skills))
        skill_col2.metric("Aapke Resume Mein Skills", len(set(resume_skills)))
        skill_col3.metric("Matched Skills", len(matched_skills))

        st.subheader("✅ Matched Skills")
        st.write(f"`{', '.join(matched_skills) or 'None'}`")
        
        st.subheader("⚠️ Missing Must-Have Skills")
        st.write(f"`{', '.join(set(jd_must_have) - set(resume_skills)) or 'None! Good job.'}`")
        
        st.subheader("⚡ Missing Good-to-Have Skills")
        st.write(f"`{', '.join(set(jd_good_to_have) - set(resume_skills)) or 'None! Good job.'}`")
        
        st.session_state.demo_resume_text = ""


