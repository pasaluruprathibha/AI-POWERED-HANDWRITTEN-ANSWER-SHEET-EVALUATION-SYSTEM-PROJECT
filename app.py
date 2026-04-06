from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import easyocr
import cv2
from textblob import TextBlob
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

app = Flask(__name__)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- GROQ LLM ----------------
llm = ChatGroq(
    api_key="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",
    model_name="llama-3.1-8b-instant"
)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# ---------------- DATABASE ----------------
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    password TEXT
                 )''')
    conn.commit()
    conn.close()

init_db()

# ---------------- ROUTES ----------------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?,?)", (username, password))
            conn.commit()
            flash("Registered successfully! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()
        if user:
            session['username'] = username
            return redirect(url_for('user_home'))
        else:
            flash("Invalid credentials!", "danger")
    return render_template('login.html')

@app.route('/user_home', methods=['GET','POST'])
def user_home():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash("No file part", "danger")
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash("No selected file", "danger")
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # ---------- OCR ----------
            image = cv2.imread(filepath)
            results = reader.readtext(image)
            raw_text = ' '.join([text for (_, text, _) in results])
            corrected_text = str(TextBlob(raw_text).correct())

            # ---------- Structured Paragraph via LLM ----------
            prompt = f"""
            Given the following text, identify the main points and structure it into a clear and concise paragraph:
            Rules:
            1. Focus on the key information and main ideas.
            2. Ensure the paragraph is coherent and well-organized.
            {corrected_text}

            Please provide only the structured paragraph, no extra explanation.
            """
            chain = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
            structured_paragraph = chain.invoke({"text": prompt})

            # ---------- Cosine similarity ----------
            def cosine_similarity(text1, text2):
                words = set(text1.split()) | set(text2.split())
                vec1 = [text1.split().count(word) for word in words]
                vec2 = [text2.split().count(word) for word in words]
                dot_product = sum(a*b for a,b in zip(vec1, vec2))
                mag1 = sum(a**2 for a in vec1)**0.5
                mag2 = sum(b**2 for b in vec2)**0.5
                return 0.0 if mag1==0 or mag2==0 else dot_product/(mag1*mag2)
            similarity_score = cosine_similarity(corrected_text, structured_paragraph)

            # ---------- Partial matching ----------
            def partial_matching_score(text1,text2):
                words1 = set(text1.split())
                words2 = set(text2.split())
                return len(words1 & words2)/len(words1) if words1 else 0.0
            partial_score = partial_matching_score(corrected_text, structured_paragraph)

            # ---------- LLM Score ----------
            score_prompt = f"""
            Please assign a numeric score between 0 and 10 to the following structured paragraph 
            based on how well it captures the essence of the original text while improving readability 
            and coherence. Respond only with the number.

            Rules:
            1. If the structured paragraph captures all main points and is well-written, score close to 10.
            2. If the para is not related to any answer, score close to 0.
            3. If the para captures some points but misses others, score between 1 and 9 based on the extent of coverage and quality.
            4. If the para is exactly correct relevant to outside knowledge, score 10.
            5. If the para is completely irrelevant, score 0.
            6. If para has "Unable to process the text" then give score direct 0.

            {structured_paragraph}
            """
            chain_score = ChatPromptTemplate.from_template("{text}") | llm | StrOutputParser()
            llm_response = chain_score.invoke({"text": score_prompt})
            
            # Extract numeric score safely
            match = re.search(r'\d+(\.\d+)?', llm_response)
            if match:
                llm_score = float(match.group())
            else:
                llm_score = 0.0

            # ---------- Final Evaluation ----------
            final_score = 0.3*similarity_score + 0.2*partial_score + 0.5*(llm_score/10)

            return render_template('result.html',
                                   raw_text=raw_text,
                                   corrected_text=corrected_text,
                                   structured_paragraph=structured_paragraph,
                                   similarity_score=similarity_score,
                                   partial_score=partial_score,
                                   llm_score=llm_score,
                                   final_score=final_score)
    return render_template('user_home.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
