import gradio as gr
import pandas as pd
import numpy as np
import sqlite3
import json
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import tempfile

# PDF processing with fallback
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

class ResumeAnalyzer:
    def __init__(self):
        self.skills = [
            'python', 'java', 'javascript', 'sql', 'mysql', 'postgresql',
            'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
            'power bi', 'tableau', 'excel', 'statistics', 'machine learning',
            'data analysis', 'data visualization', 'react', 'angular', 'django',
            'flask', 'nodejs', 'mongodb', 'aws', 'azure', 'docker', 'git',
            'tensorflow', 'pytorch', 'keras', 'opencv', 'nltk', 'spacy',
            'spark', 'hadoop', 'kafka', 'r', 'sas', 'spss', 'business intelligence'
        ]
        
        self.education_keywords = [
            'bachelor', 'b.tech', 'btech', 'b.e', 'be', 'b.sc', 'bsc',
            'master', 'm.tech', 'mtech', 'm.sc', 'msc', 'mba', 'phd'
        ]
        
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect('resume_analysis.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    relevance_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    found_skills TEXT,
                    missing_skills TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF"""
        if not PDF_AVAILABLE:
            return "PDF processing not available. Please convert to text file."
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            return f"Error reading PDF: {e}"
    
    def extract_skills(self, text):
        """Extract skills from text"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def extract_education(self, text):
        """Extract education from text"""
        if not text:
            return []
        
        text_lower = text.lower()
        found_education = []
        
        for edu in self.education_keywords:
            if edu in text_lower:
                found_education.append(edu)
        
        return found_education
    
    def analyze_single_resume(self, resume_file, job_description):
        """Analyze a single resume"""
        try:
            if not resume_file or not job_description.strip():
                return "Error: Please provide both resume file and job description."
            
            # Extract text from file
            resume_text = ""
            if resume_file.name.endswith('.pdf'):
                resume_text = self.extract_text_from_pdf(resume_file.name)
            elif resume_file.name.endswith(('.txt', '.text')):
                with open(resume_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    resume_text = f.read()
            else:
                return "Error: Please upload PDF or TXT files only."
            
            if not resume_text.strip():
                return "Error: Could not extract text from file."
            
            # Analyze resume
            resume_skills = self.extract_skills(resume_text)
            resume_education = self.extract_education(resume_text)
            jd_skills = self.extract_skills(job_description)
            
            # Calculate scores
            if jd_skills:
                skill_matches = len(set(resume_skills) & set(jd_skills))
                hard_score = (skill_matches / len(jd_skills)) * 70
            else:
                hard_score = 50
            
            education_score = 20 if resume_education else 0
            hard_score = min(hard_score + education_score, 100)
            
            # Semantic score
            try:
                corpus = [resume_text, job_description]
                tfidf_matrix = self.tfidf.fit_transform(corpus)
                semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
            except:
                semantic_score = 50
            
            # Final score
            final_score = (hard_score * 0.7) + (semantic_score * 0.3)
            
            # Verdict
            if final_score >= 75:
                verdict = "High"
            elif final_score >= 50:
                verdict = "Medium"
            else:
                verdict = "Low"
            
            # Missing skills
            missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
            
            # Save to database
            self.save_result(
                os.path.basename(resume_file.name),
                final_score,
                verdict,
                resume_skills,
                missing_skills
            )
            
            # Format output
            result = f"""
## Analysis Results

**File:** {os.path.basename(resume_file.name)}
**Relevance Score:** {final_score:.1f}/100
**Verdict:** {verdict}
**Hard Match Score:** {hard_score:.1f}/100
**Semantic Score:** {semantic_score:.1f}/100

### Found Skills ({len(resume_skills)})
{', '.join(resume_skills[:15]) if resume_skills else 'None identified'}

### Missing Skills ({len(missing_skills)})
{', '.join(missing_skills[:15]) if missing_skills else 'All required skills found!'}

### Improvement Suggestions
- {'Consider gaining experience in: ' + ', '.join(missing_skills[:5]) if missing_skills else 'Great skill match!'}
- Add relevant projects showcasing technical abilities
- Include certifications related to job requirements
- Quantify achievements with specific metrics
            """
            
            return result
            
        except Exception as e:
            return f"Analysis Error: {str(e)}"
    
    def analyze_bulk_resumes(self, resume_files, job_description, progress=gr.Progress()):
        """Analyze multiple resumes"""
        if not resume_files or not job_description.strip():
            return "Error: Please provide resume files and job description."
        
        results = []
        total_files = len(resume_files)
        
        for i, resume_file in enumerate(resume_files):
            progress((i + 1) / total_files, f"Processing {resume_file.name}...")
            
            try:
                # Extract text
                resume_text = ""
                if resume_file.name.endswith('.pdf') and PDF_AVAILABLE:
                    resume_text = self.extract_text_from_pdf(resume_file.name)
                elif resume_file.name.endswith(('.txt', '.text')):
                    with open(resume_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        resume_text = f.read()
                else:
                    continue
                
                if not resume_text.strip():
                    continue
                
                # Quick analysis for bulk
                resume_skills = self.extract_skills(resume_text)
                jd_skills = self.extract_skills(job_description)
                
                # Simple scoring for bulk
                if jd_skills:
                    skill_matches = len(set(resume_skills) & set(jd_skills))
                    score = (skill_matches / len(jd_skills)) * 100
                else:
                    score = 50
                
                if score >= 75:
                    verdict = "High"
                elif score >= 50:
                    verdict = "Medium"
                else:
                    verdict = "Low"
                
                results.append({
                    'filename': os.path.basename(resume_file.name),
                    'score': round(score, 1),
                    'verdict': verdict,
                    'found_skills': len(resume_skills),
                    'missing_skills': len([s for s in jd_skills if s not in resume_skills])
                })
                
                # Save to database
                self.save_result(
                    os.path.basename(resume_file.name),
                    score,
                    verdict,
                    resume_skills,
                    [s for s in jd_skills if s not in resume_skills]
                )
                
            except Exception as e:
                print(f"Error processing {resume_file.name}: {e}")
                continue
        
        if not results:
            return "No files could be processed successfully."
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Create summary
        avg_score = np.mean([r['score'] for r in results])
        high_count = len([r for r in results if r['verdict'] == 'High'])
        
        # Format results table
        df = pd.DataFrame(results)
        
        summary = f"""
## Bulk Analysis Results

**Total Processed:** {len(results)} resumes  
**Average Score:** {avg_score:.1f}/100  
**High Matches:** {high_count}  
**Top Score:** {max([r['score'] for r in results]):.1f}/100

### Rankings
"""
        
        for i, result in enumerate(results[:10], 1):
            summary += f"{i}. **{result['filename']}** - {result['score']}/100 ({result['verdict']})\n"
        
        return summary, df
    
    def save_result(self, filename, score, verdict, found_skills, missing_skills):
        """Save result to database"""
        try:
            conn = sqlite3.connect('resume_analysis.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_results 
                (filename, relevance_score, verdict, found_skills, missing_skills, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                score,
                verdict,
                json.dumps(found_skills),
                json.dumps(missing_skills),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Save error: {e}")
    
    def get_dashboard_data(self):
        """Get dashboard statistics"""
        try:
            conn = sqlite3.connect('resume_analysis.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM analysis_results ORDER BY timestamp DESC LIMIT 20')
            rows = cursor.fetchall()
            
            if not rows:
                return "No analysis data available. Process some resumes first!"
            
            # Calculate statistics
            scores = [row[2] for row in rows]
            verdicts = [row[3] for row in rows]
            
            total_resumes = len(rows)
            avg_score = np.mean(scores)
            high_count = len([v for v in verdicts if v == 'High'])
            medium_count = len([v for v in verdicts if v == 'Medium'])
            low_count = len([v for v in verdicts if v == 'Low'])
            
            # Create recent results table
            recent_data = []
            for row in rows[:10]:
                recent_data.append({
                    'Filename': row[1],
                    'Score': f"{row[2]:.1f}/100",
                    'Verdict': row[3],
                    'Timestamp': row[6][:16]
                })
            
            df = pd.DataFrame(recent_data)
            
            summary = f"""
## Dashboard Overview

### Key Metrics
- **Total Resumes Analyzed:** {total_resumes}
- **Average Score:** {avg_score:.1f}/100
- **High Matches:** {high_count}
- **Medium Matches:** {medium_count}
- **Low Matches:** {low_count}

### Distribution
- High (75-100): {(high_count/total_resumes)*100:.1f}%
- Medium (50-74): {(medium_count/total_resumes)*100:.1f}%
- Low (0-49): {(low_count/total_resumes)*100:.1f}%
            """
            
            conn.close()
            return summary, df
            
        except Exception as e:
            return f"Dashboard Error: {str(e)}", pd.DataFrame()

# Initialize analyzer
analyzer = ResumeAnalyzer()

# Job description templates
JD_TEMPLATES = {
    "Data Science Role": """Data Scientist Position

We are looking for a Data Scientist to join our analytics team.

Required Skills:
- Python programming (Pandas, NumPy, Matplotlib, Seaborn)
- SQL database knowledge (MySQL, PostgreSQL)
- Machine learning fundamentals
- Statistics and statistical analysis
- Data visualization tools (Power BI, Tableau)
- Experience with data cleaning and preprocessing

Qualifications:
- Bachelor's degree in Computer Science, Statistics, or related field
- 2+ years of data analysis experience
- Strong analytical and problem-solving skills""",

    "Software Engineer Role": """Software Engineering Position

We are seeking a Software Engineer for our development team.

Required Skills:
- Python, Java, or JavaScript programming
- Web development frameworks (React, Angular, Django, Flask)
- Database knowledge (MySQL, PostgreSQL, MongoDB)
- Version control with Git
- REST API development
- HTML, CSS, JavaScript

Qualifications:
- Bachelor's degree in Computer Science or related field
- 1+ years of software development experience
- Strong problem-solving abilities""",

    "Business Analyst Role": """Business Analyst Position

Looking for a Business Analyst to drive data-driven decision making.

Required Skills:
- SQL for data extraction and analysis
- Excel advanced functions and pivot tables
- Power BI or Tableau for data visualization
- Business intelligence and analytics
- Data interpretation and reporting
- Statistical analysis fundamentals

Qualifications:
- Bachelor's degree in Business, Engineering, or related field
- 2+ years of business analysis experience
- Strong communication and presentation skills"""
}

def load_template(template_name):
    """Load job description template"""
    return JD_TEMPLATES.get(template_name, "")

# Create Gradio interface
with gr.Blocks(title="Resume Relevance Check System", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🎯 Automated Resume Relevance Check System
    ## AI-powered resume evaluation against job requirements | Innomatics Research Labs
    """)
    
    with gr.Tabs():
        # Single Analysis Tab
        with gr.TabItem("📋 Single Analysis"):
            gr.Markdown("### Analyze individual resume against job description")
            
            with gr.Row():
                with gr.Column(scale=1):
                    resume_file = gr.File(
                        label="Upload Resume", 
                        file_types=[".pdf", ".txt"],
                        type="filepath"
                    )
                
                with gr.Column(scale=2):
                    template_dropdown = gr.Dropdown(
                        choices=list(JD_TEMPLATES.keys()),
                        label="Quick Templates (Optional)",
                        value=None
                    )
                    
                    job_description = gr.Textbox(
                        label="Job Description",
                        placeholder="Enter the complete job description...",
                        lines=8
                    )
            
            analyze_btn = gr.Button("🔍 Analyze Resume", variant="primary", size="lg")
            
            with gr.Row():
                analysis_output = gr.Markdown(label="Analysis Results")
            
            # Template loading
            template_dropdown.change(
                fn=load_template,
                inputs=template_dropdown,
                outputs=job_description
            )
            
            # Analysis function
            analyze_btn.click(
                fn=analyzer.analyze_single_resume,
                inputs=[resume_file, job_description],
                outputs=analysis_output
            )
        
        # Bulk Processing Tab
        with gr.TabItem("📦 Bulk Processing"):
            gr.Markdown("### Process multiple resumes simultaneously")
            
            bulk_jd_template = gr.Dropdown(
                choices=list(JD_TEMPLATES.keys()),
                label="Quick Templates (Optional)"
            )
            
            bulk_job_description = gr.Textbox(
                label="Job Description for All Resumes",
                placeholder="Enter the job description...",
                lines=6
            )
            
            bulk_files = gr.File(
                label="Upload Multiple Resumes",
                file_count="multiple",
                file_types=[".pdf", ".txt"],
                type="filepath"
            )
            
            bulk_analyze_btn = gr.Button("⚡ Process All Resumes", variant="primary", size="lg")
            
            bulk_summary = gr.Markdown(label="Bulk Analysis Summary")
            bulk_table = gr.Dataframe(label="Detailed Results")
            
            # Template loading for bulk
            bulk_jd_template.change(
                fn=load_template,
                inputs=bulk_jd_template,
                outputs=bulk_job_description
            )
            
            # Bulk analysis
            bulk_analyze_btn.click(
                fn=analyzer.analyze_bulk_resumes,
                inputs=[bulk_files, bulk_job_description],
                outputs=[bulk_summary, bulk_table]
            )
        
        # Dashboard Tab
        with gr.TabItem("📊 Dashboard"):
            gr.Markdown("### Analysis statistics and recent results")
            
            refresh_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
            
            dashboard_summary = gr.Markdown(value="Click refresh to load dashboard data")
            dashboard_table = gr.Dataframe(label="Recent Analysis Results")
            
            # Dashboard refresh
            refresh_btn.click(
                fn=analyzer.get_dashboard_data,
                outputs=[dashboard_summary, dashboard_table]
            )
    
    gr.Markdown("""
    ---
    **Built for Code4EdTech Hackathon 2025 | Innomatics Research Labs**  
    *Powered by AI • Python • Gradio • Machine Learning*
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL for demo
        debug=False
    )