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
import threading
import signal
import sys

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

# Disable signal handlers for threaded environments
class CustomFlask(Flask):
    def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
        # Disable signal handlers to avoid threading issues
        options.setdefault('use_reloader', False)
        options.setdefault('threaded', True)
        
        # Remove signal handling in non-main threads
        if threading.current_thread() is not threading.main_thread():
            signal.signal = lambda *args: None
        
        super().run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)

app = CustomFlask(__name__)
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
            'jenkins', 'linux', 'bash', 'html', 'css', 'bootstrap', 'jquery',
            'r programming', 'sas', 'spss', 'alteryx', 'looker', 'qlik',
            'business intelligence', 'data mining', 'deep learning', 'ai',
            'artificial intelligence', 'natural language processing', 'nlp'
        ]
        
        self.education_keywords = [
            'bachelor', 'b.tech', 'btech', 'b.e', 'be', 'b.sc', 'bsc',
            'master', 'm.tech', 'mtech', 'm.sc', 'msc', 'mba', 'phd',
            'graduation', 'post graduation', 'degree', 'diploma'
        ]
        
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        
        # Initialize database
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect('resume_analysis.db', check_same_thread=False)
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
        except Exception as e:
            print(f"Database initialization error: {e}")
    
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
        if not text:
            return ""
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
        
        return list(set(found_skills))  # Remove duplicates
    
    def extract_education(self, text):
        """Extract education information"""
        text = self.clean_text(text)
        found_education = []
        
        for edu in self.education_keywords:
            if edu in text:
                found_education.append(edu)
        
        return list(set(found_education))  # Remove duplicates
    
    def calculate_hard_score(self, resume_skills, resume_education, jd_skills):
        """Calculate keyword-based score"""
        if not jd_skills:
            return 50
        
        # Skill matching (70% of hard score)
        skill_matches = len(set(resume_skills) & set(jd_skills))
        skill_score = (skill_matches / len(jd_skills)) * 70
        
        # Education bonus (30% of hard score)
        education_score = 30 if resume_education else 0
        
        return min(skill_score + education_score, 100)
    
    def calculate_semantic_score(self, resume_text, jd_text):
        """Calculate semantic similarity"""
        try:
            if not resume_text.strip() or not jd_text.strip():
                return 50
            
            corpus = [resume_text, jd_text]
            tfidf_matrix = self.tfidf.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return min(similarity * 100, 100)
        except Exception as e:
            print(f"Semantic score calculation error: {e}")
            return 50
    
    def analyze_resume(self, resume_text, jd_text, filename=""):
        """Main analysis function"""
        try:
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
                'found_skills': resume_skills[:20],  # Limit to prevent overflow
                'missing_skills': missing_skills[:20],  # Limit to prevent overflow
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to database
            self.save_to_database(result, jd_text)
            
            return result
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'filename': filename,
                'relevance_score': 0,
                'hard_score': 0,
                'semantic_score': 0,
                'verdict': 'Error',
                'found_skills': [],
                'missing_skills': [],
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def save_to_database(self, result, jd_text):
        """Save analysis result to database"""
        try:
            conn = sqlite3.connect('resume_analysis.db', check_same_thread=False)
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
        except Exception as e:
            print(f"Database save error: {e}")
    
    def get_dashboard_data(self):
        """Get dashboard statistics"""
        try:
            conn = sqlite3.connect('resume_analysis.db', check_same_thread=False)
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
            avg_score = np.mean([r['relevance_score'] for r in results]) if results else 0
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
            
        except Exception as e:
            print(f"Dashboard data error: {e}")
            return {
                'total_resumes': 0,
                'avg_score': 0,
                'high_count': 0,
                'medium_count': 0,
                'low_count': 0,
                'recent_results': []
            }

# Initialize analyzer
analyzer = ResumeAnalyzer()

# Route handlers
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
        
        # Validate file size
        if resume_file.content_length and resume_file.content_length > 16 * 1024 * 1024:
            return jsonify({'error': 'File too large (max 16MB)'}), 400
        
        # Save uploaded file
        filename = secure_filename(resume_file.filename)
        if not filename:
            filename = 'unnamed_resume.txt'
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        resume_file.save(file_path)
        
        # Extract text
        resume_text = ""
        try:
            if filename.lower().endswith('.pdf') and PDF_AVAILABLE:
                resume_text = analyzer.extract_text_from_pdf(file_path)
            elif filename.lower().endswith(('.txt', '.text')):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
            else:
                return jsonify({'error': 'Unsupported file format. Please use PDF or TXT files.'}), 400
        except Exception as e:
            return jsonify({'error': f'Error reading file: {str(e)}'}), 400
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
        
        if not resume_text.strip():
            return jsonify({'error': 'Could not extract text from file or file is empty'}), 400
        
        # Analyze resume
        result = analyzer.analyze_resume(resume_text, jd_text, filename)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/bulk-analyze', methods=['POST'])
def api_bulk_analyze():
    """API endpoint for bulk analysis"""
    try:
        if 'resumes' not in request.files or 'job_description' not in request.form:
            return jsonify({'error': 'Missing files or job description'}), 400
        
        files = request.files.getlist('resumes')
        jd_text = request.form['job_description']
        
        if len(files) > 50:
            return jsonify({'error': 'Maximum 50 files allowed per batch'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            try:
                filename = secure_filename(file.filename)
                if not filename:
                    filename = f'unnamed_resume_{len(results)}.txt'
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Extract text
                resume_text = ""
                if filename.lower().endswith('.pdf') and PDF_AVAILABLE:
                    resume_text = analyzer.extract_text_from_pdf(file_path)
                elif filename.lower().endswith(('.txt', '.text')):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        resume_text = f.read()
                
                # Clean up immediately
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                if resume_text.strip():
                    result = analyzer.analyze_resume(resume_text, jd_text, filename)
                    results.append(result)
                
            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                continue
        
        return jsonify({
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/dashboard')
def api_dashboard():
    """API endpoint for dashboard data"""
    try:
        data = analyzer.get_dashboard_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/export')
def api_export():
    """Export results as CSV"""
    try:
        conn = sqlite3.connect('resume_analysis.db', check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT filename, relevance_score, hard_score, semantic_score, 
                   verdict, timestamp 
            FROM analysis_results 
            ORDER BY timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return jsonify({'error': 'No data to export'}), 404
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Filename', 'Relevance Score', 'Hard Score', 'Semantic Score', 'Verdict', 'Timestamp'])
        
        # Write data
        for row in rows:
            writer.writerow(row)
        
        # Prepare response
        output.seek(0)
        csv_content = output.getvalue()
        output.close()
        
        return send_file(
            io.BytesIO(csv_content.encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'resume_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': f'Export error: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run with proper configuration for development
    app.run(
        debug=False,  # Disable debug to avoid signal issues
        host='0.0.0.0', 
        port=5000,
        use_reloader=False,  # Disable reloader to avoid threading issues
        threaded=True
    )
