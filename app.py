from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import sqlite3
import re
from datetime import datetime
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import csv

# PDF processing
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# NLTK with fallback
try:
    import nltk
    from nltk.corpus import stopwords
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
except ImportError:
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'resume-relevance-system-key'

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ResumeAnalyzer:
    def __init__(self):
        # Comprehensive skill keywords database
        self.skills = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
            'power bi', 'tableau', 'excel', 'statistics', 'machine learning',
            'data analysis', 'data visualization', 'react', 'angular', 'django',
            'flask', 'nodejs', 'mongodb', 'aws', 'azure', 'docker', 'git',
            'tensorflow', 'pytorch', 'keras', 'opencv', 'nltk', 'spacy',
            'spark', 'hadoop', 'kafka', 'elasticsearch', 'redis', 'kubernetes',
            'jenkins', 'linux', 'bash', 'html', 'css', 'bootstrap', 'jquery'
        ]
        
        self.education_keywords = [
            'bachelor', 'b.tech', 'btech', 'b.e', 'be', 'b.sc', 'bsc',
            'master', 'm.tech', 'mtech', 'm.sc', 'msc', 'mba', 'phd'
        ]
        
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect('resume_analysis.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                hard_score REAL NOT NULL,
                semantic_score REAL NOT NULL,
                verdict TEXT NOT NULL,
                found_skills TEXT,
                missing_skills TEXT,
                job_description TEXT,
                timestamp TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        if not PDF_AVAILABLE:
            return ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip().lower()
    
    def extract_skills(self, text):
        """Extract skills from text"""
        text = self.clean_text(text)
        found_skills = []
        
        for skill in self.skills:
            if skill in text:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_education(self, text):
        """Extract education information"""
        text = self.clean_text(text)
        found_education = []
        
        for edu in self.education_keywords:
            if edu in text:
                found_education.append(edu)
        
        return found_education
    
    def calculate_hard_score(self, resume_skills, resume_education, jd_skills):
        """Calculate keyword-based score"""
        if not jd_skills:
            return 50
        
        skill_matches = len(set(resume_skills) & set(jd_skills))
        skill_score = (skill_matches / len(jd_skills)) * 70
        
        education_score = 20 if resume_education else 0
        
        return min(skill_score + education_score, 100)
    
    def calculate_semantic_score(self, resume_text, jd_text):
        """Calculate semantic similarity"""
        try:
            corpus = [resume_text, jd_text]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity * 100
        except:
            return 50
    
    def analyze_resume(self, resume_text, jd_text, filename=""):
        """Main analysis function"""
        resume_skills = self.extract_skills(resume_text)
        resume_education = self.extract_education(resume_text)
        jd_skills = self.extract_skills(jd_text)
        
        hard_score = self.calculate_hard_score(resume_skills, resume_education, jd_skills)
        semantic_score = self.calculate_semantic_score(resume_text, jd_text)
        
        final_score = (hard_score * 0.7) + (semantic_score * 0.3)
        
        if final_score >= 75:
            verdict = "High"
        elif final_score >= 50:
            verdict = "Medium"
        else:
            verdict = "Low"
        
        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
        
        result = {
            'filename': filename,
            'relevance_score': round(final_score, 2),
            'hard_score': round(hard_score, 2),
            'semantic_score': round(semantic_score, 2),
            'verdict': verdict,
            'found_skills': resume_skills,
            'missing_skills': missing_skills,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to database
        self.save_to_database(result, jd_text)
        
        return result
    
    def save_to_database(self, result, jd_text):
        """Save analysis result to database"""
        conn = sqlite3.connect('resume_analysis.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (filename, relevance_score, hard_score, semantic_score, verdict, 
             found_skills, missing_skills, job_description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['filename'],
            result['relevance_score'],
            result['hard_score'],
            result['semantic_score'],
            result['verdict'],
            json.dumps(result['found_skills']),
            json.dumps(result['missing_skills']),
            jd_text[:500],  # Store first 500 chars of JD
            result['timestamp']
        ))
        
        conn.commit()
        conn.close()
    
    def get_dashboard_data(self):
        """Get dashboard statistics"""
        conn = sqlite3.connect('resume_analysis.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM analysis_results ORDER BY timestamp DESC LIMIT 100')
        rows = cursor.fetchall()
        
        if not rows:
            return {
                'total_resumes': 0,
                'avg_score': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'recent_results': []
            }
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'filename': row[1],
                'relevance_score': row[2],
                'hard_score': row[3],
                'semantic_score': row[4],
                'verdict': row[5],
                'found_skills': json.loads(row[6]) if row[6] else [],
                'missing_skills': json.loads(row[7]) if row[7] else [],
                'timestamp': row[9]
            })
        
        # Calculate statistics
        total_resumes = len(results)
        avg_score = np.mean([r['relevance_score'] for r in results])
        high_count = len([r for r in results if r['verdict'] == 'High'])
        medium_count = len([r for r in results if r['verdict'] == 'Medium'])
        low_count = len([r for r in results if r['verdict'] == 'Low'])
        
        conn.close()
        
        return {
            'total_resumes': total_resumes,
            'avg_score': round(avg_score, 2),
            'high_count': high_count,
            'medium_count': medium_count,
            'low_count': low_count,
            'recent_results': results[:10]
        }

# Initialize analyzer
analyzer = ResumeAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/analyze')
def analyze_page():
    """Single analysis page"""
    return render_template('analyze.html')

@app.route('/bulk')
def bulk_page():
    """Bulk analysis page"""
    return render_template('bulk.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for single resume analysis"""
    try:
        if 'resume' not in request.files or 'job_description' not in request.form:
            return jsonify({'error': 'Missing resume file or job description'}), 400
        
        resume_file = request.files['resume']
        jd_text = request.form['job_description']
        
        if resume_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(resume_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)
        
        # Extract text
        if filename.lower().endswith('.pdf'):
            resume_text = analyzer.extract_text_from_pdf(file_path)
        elif filename.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                resume_text = f.read()
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        if not resume_text.strip():
            return jsonify({'error': 'Could not extract text from file'}), 400
        
        # Analyze resume
        result = analyzer.analyze_resume(resume_text, jd_text, filename)
        
        # Clean up uploaded file
        os.remove(file_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk-analyze', methods=['POST'])
def api_bulk_analyze():
    """API endpoint for bulk analysis"""
    try:
        if 'resumes' not in request.files or 'job_description' not in request.form:
            return jsonify({'error': 'Missing files or job description'}), 400
        
        files = request.files.getlist('resumes')
        jd_text = request.form['job_description']
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text
            try:
                if filename.lower().endswith('.pdf'):
                    resume_text = analyzer.extract_text_from_pdf(file_path)
                elif filename.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        resume_text = f.read()
                else:
                    continue
                
                if resume_text.strip():
                    result = analyzer.analyze_resume(resume_text, jd_text, filename)
                    results.append(result)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
            # Clean up
            os.remove(file_path)
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    try:
        data = analyzer.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def api_export():
    """Export results as CSV"""
    try:
        conn = sqlite3.connect('resume_analysis.db')
        df = pd.read_sql_query('''
            SELECT filename, relevance_score, hard_score, semantic_score, 
                   verdict, timestamp 
            FROM analysis_results 
            ORDER BY timestamp DESC
        ''', conn)
        conn.close()
        
        if df.empty:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'resume_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)