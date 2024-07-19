from flask import Flask, request, render_template
import pickle
from PyPDF2 import PdfReader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

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
        recommendation = job_recommendation(text).capitalize()

        return render_template("resume.html", text=text, recommendation=recommendation)

    else:
        pass


if __name__ == "__main__":
    app.run(debug=True)