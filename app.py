#!/usr/bin/env python3
"""
Simple startup script for Resume Relevance Check System
Run this file to launch the application
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def setup_nltk():
    """Setup NLTK data"""
    print("Setting up NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK data downloaded")
    except Exception as e:
        print(f"NLTK setup warning: {e}")

def main():
    print("🎯 Resume Relevance Check System - Starting...")
    print("=" * 50)
    
    # Check if requirements file exists
    if not os.path.exists("requirements.txt"):
        print("Creating requirements.txt...")
        with open("requirements.txt", "w") as f:
            f.write("""gradio>=4.0.0
pandas>=2.0.0
numpy>=1.25.0
scikit-learn>=1.3.0
nltk>=3.8
PyPDF2>=3.0.0""")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please install manually:")
        print("pip install gradio pandas numpy scikit-learn nltk PyPDF2")
        return
    
    # Setup NLTK
    setup_nltk()
    
    # Import and run the main application
    print("\n🚀 Launching application...")
    print("📱 The app will open in your browser automatically")
    print("🔗 If not, check the console for the local URL")
    print("🌐 A public shareable URL will also be provided")
    print("\n" + "=" * 50)
    
    try:
        # Import the main app (assumes it's in the same directory)
        from gradio_solution import app
        
        # Launch with public sharing enabled
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            debug=False,
            show_error=True
        )
        
    except ImportError:
        print("Error: Could not import the main application.")
        print("Make sure the main application file is named 'gradio_solution.py'")
    except Exception as e:
        print(f"Error launching application: {e}")
        print("\nTrying alternative launch method...")
        
        # Alternative: Run as subprocess
        try:
            subprocess.run([sys.executable, "gradio_solution.py"])
        except Exception as e2:
            print(f"Alternative launch failed: {e2}")
            print("Please run the application manually: python gradio_solution.py")

if __name__ == "__main__":
    main()
