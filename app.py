from flask import Flask, request, render_template
import pickle
from PyPDF2 import PdfReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
# from pdfminer.high_level import extract_text
from lists import skills_list, education_keywords

# rf_classifier_categorization = pickle.load(open("models/rf_classifier_categorization.pkl", "rb"))
# tfidf_vectorizer_categorization = pickle.load(open("models/tfidf_vectorizer_categorization.pkl", "rb"))
rf_classifier_job_recommendation = pickle.load(open("models/rf_classifier_job_recommendation6500.pkl", "rb"))
tfidf_vectorizer_job_recommendation = pickle.load(open("models/tfidf_vectorizer_job_recommendation6500.pkl", "rb"))


def pdf_to_text(file):
    reader = PdfReader(file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text


def cleaning(string: str):
    lemmatizer = WordNetLemmatizer()
    output = re.sub("[^\w]", " ", string)
    output = re.sub("\s+", " ", output)
    output = output.strip().lower().split()
    output = [lemmatizer.lemmatize(word) for word in output if word not in stopwords.words("english")]
    output = " ".join(output)
    
    return output

#---------------------------------------------- Extracting Information ----------------------------------------
# def extract_text_from_pdf(pdf_path):
#     return extract_text(pdf_path)

def extract_name_from_resume(text):
    name = None

    # Use regex pattern to find a potential name
    pattern = r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)"
    match = re.search(pattern, text)
    if match:
        name = match.group()

    return name

def extract_contact_number_from_resume(text):
    contact_number = None

    # Use regex pattern to find a potential contact number
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    if match:
        contact_number = match.group()

    return contact_number

def extract_email_from_resume(text):
    email = None

    # Use regex pattern to find a potential email address
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    if match:
        email = match.group()

    return email

def extract_skills_from_resume(text, skills_list):
    skills = []

    for skill in skills_list:
        pattern = r"\b{}\b".format(re.escape(skill))
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            skills.append(skill)

    return skills

def extract_education_from_resume(text, education_keywords):
    education = []
    # List of education keywords to match against

    for keyword in education_keywords:
        pattern = r"(?i)\b{}\b".format(re.escape(keyword))
        match = re.search(pattern, text)
        if match:
            education.append(match.group())

    return education


def extract_information(file):
    text = pdf_to_text(file)
    name = extract_name_from_resume(text)
    contact_no = extract_contact_number_from_resume(text)
    email = extract_email_from_resume(text)
    skills = extract_skills_from_resume(text, skills_list)
    education = extract_education_from_resume(text, education_keywords)

    return [name, contact_no, email, skills, education]





# def predict_category(resume_text):
#     resume = cleaning(resume_text)
#     resume = tfidf_vectorizer_categorization.transform([resume])
#     predicted = rf_classifier_categorization.predict(resume[0])
    
#     return predicted[0]


def job_recommendation(resume_text):
    resume = cleaning(resume_text)
    resume = tfidf_vectorizer_job_recommendation.transform([resume])
    recommended_job = rf_classifier_job_recommendation.predict(resume[0])
    
    return recommended_job[0]


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("resume.html")


@app.route("/pred", methods=["POST"])
def pred():
    if "resume" in request.files:
        file = request.files["resume"]
        filename = file.filename

        text = ""
        if filename.endswith(".pdf"):
            text = pdf_to_text(file)
        elif filename.endswith(".txt"):
            text = file.read().decode("utf-8")

        # prediction = predict_category(text).capitalize()
        recommendation = job_recommendation(text)

        information_extracted = extract_information(file)

        return render_template("resume.html",
                               text=text,
                               recommended_job=recommendation,
                               name=information_extracted[0],
                               phone=information_extracted[1],
                               email=information_extracted[2],
                               extracted_skills=information_extracted[3],
                               extracted_education=information_extracted[4]
                               )

    else:
        pass


if __name__ == "__main__":
    app.run(debug=True)