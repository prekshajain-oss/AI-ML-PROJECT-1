AI BASED RESUME SCREENING SYSTEM   
import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 AI Resume Screening System \n Make your resume more efiicient for your job interviews and placements")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Job Description Input
job_description = st.text_area(" Enter Your Job Description")

# Upload Multiple Resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    resume_texts = []
    resume_names = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resume_texts.append(text)
        resume_names.append(file.name)

    documents = [job_description] + resume_texts

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Calculate similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    results = list(zip(resume_names, similarity_scores))
    results.sort(key=lambda x: x[1], reverse=True)

    st.subheader("📊 Resume Ranking")

    for name, score in results:
        st.write(f"📄 {name} — Match Score: {round(score*100, 2)}%")

