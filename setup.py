#!/usr/bin/env python3
"""
Setup script for Disaster Preparedness RAG Pipeline
Automates the complete pipeline setup and execution
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from dotenv import load_dotenv

def run_command(command, description):
    """Run a shell command with error handling"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return None

def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️  .env file not found. Please copy .env.example to .env and configure your API keys")
        return False
    
    # Load and check environment variables
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        print("⚠️  GROQ_API_KEY not found in .env file")
        return False
    print("✅ Environment variables configured")
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def check_qdrant():
    """Check if Qdrant is running"""
    print("\n🔍 Checking Qdrant connection...")
    try:
        import requests
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
    except:
        pass
    
    print("⚠️  Qdrant not detected. Starting local Qdrant instance...")
    
    # Try to start Qdrant with Docker
    docker_result = run_command(
        "docker run -d -p 6333:6333 --name qdrant-disaster-prep qdrant/qdrant",
        "Starting Qdrant with Docker"
    )
    
    if docker_result is not None:
        print("⏳ Waiting for Qdrant to start...")
        time.sleep(10)
        
        try:
            response = requests.get("http://localhost:6333/collections", timeout=10)
            if response.status_code == 200:
                print("✅ Qdrant started successfully")
                return True
        except:
            pass
    
    print("❌ Could not start Qdrant. Please install Docker or run Qdrant manually")
    print("   Docker: docker run -p 6333:6333 qdrant/qdrant")
    print("   Manual: Download from https://github.com/qdrant/qdrant/releases")
    return False

def run_scraper():
    """Run the web scraper"""
    print("\n🕷️ Starting web scraping...")
    result = run_command("python scraper.py", "Web scraping from trusted sources")
    
    if result is not None:
        # Check if data was created
        data_file = Path("data/scraped_disaster_prep_data.json")
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Scraped {len(data)} documents")
            return True
    
    print("❌ Web scraping failed")
    return False

def run_embedding_processing():
    """Process documents and create embeddings"""
    print("\n🧠 Processing documents and creating embeddings...")
    result = run_command("python process_embeddings.py", "Creating embeddings and uploading to Qdrant")
    
    if result is not None:
        print("✅ Embedding processing completed")
        return True
    
    print("❌ Embedding processing failed")
    return False

def start_app():
    """Start the Streamlit application"""
    print("\n🚀 Starting Streamlit application...")
    print("   The app will open in your web browser at http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    
    try:
        subprocess.run("streamlit run app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main setup function"""
    print("🚨 Disaster Preparedness RAG Pipeline Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above and try again.")
        return
    
    # Install dependencies
    if install_dependencies() is None:
        print("\n❌ Failed to install dependencies")
        return
    
    # Check/start Qdrant
    if not check_qdrant():
        print("\n❌ Qdrant setup failed")
        return
    
    # Ask user what to do
    print("\n📋 Setup Options:")
    print("1. Full setup (scrape data + create embeddings + start app)")
    print("2. Skip scraping (use existing data + create embeddings + start app)")
    print("3. Skip data processing (start app only)")
    print("4. Run scraper only")
    print("5. Run embedding processing only")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == "1":
        # Full setup
        if run_scraper() and run_embedding_processing():
            input("\n✅ Setup completed! Press Enter to start the application...")
            start_app()
    
    elif choice == "2":
        # Skip scraping
        data_file = Path("data/scraped_disaster_prep_data.json")
        if not data_file.exists():
            print("❌ No existing data found. Please run option 1 or 4 first.")
            return
        
        if run_embedding_processing():
            input("\n✅ Processing completed! Press Enter to start the application...")
            start_app()
    
    elif choice == "3":
        # Start app only
        print("Starting application directly...")
        start_app()
    
    elif choice == "4":
        # Run scraper only
        run_scraper()
    
    elif choice == "5":
        # Run embedding processing only
        run_embedding_processing()
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
