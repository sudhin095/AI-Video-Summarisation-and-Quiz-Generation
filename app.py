# FREE Version: AI Video Summarization & Quiz Generation
# app_free.py - Completely FREE alternative to paid APIs

import streamlit as st
import os
import tempfile
import json
import random
import re
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Educational Assistant (FREE)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .free-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 10px 20px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;}
    .warning-box {background: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid #ffc107;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="free-badge">✨ 100% FREE - No API Costs ✨</div>', unsafe_allow_html=True)

# Session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'quiz_responses' not in st.session_state:
    st.session_state.quiz_responses = {}

# ============ VALIDATION FUNCTIONS ============

def validate_url(url):
    """Validate URL format"""
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def validate_text(text):
    """Validate text input"""
    return len(text.split()) >= 50

# ============ FREE VIDEO PROCESSING ============

def process_video_url_free(url):
    """
    Download video from URL using yt-dlp (FREE)
    Extract audio and transcribe using Vosk (FREE, offline)
    """
    try:
        import yt_dlp
        from moviepy.editor import VideoFileClip
        
        st.info("📥 Downloading video (this may take a minute)...")
        
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(tempfile.gettempdir(), '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
        
        st.info("🎙️ Extracting audio from video...")
        
        # Extract audio
        audio_path = os.path.join(tempfile.gettempdir(), 'temp_audio.wav')
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        # Use Vosk for FREE offline speech-to-text
        text = transcribe_audio_vosk(audio_path)
        
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return text
        
    except Exception as e:
        raise Exception(f"Error: {str(e)}")

def transcribe_audio_vosk(audio_path):
    """
    FREE Speech-to-Text using Vosk (offline, no API key needed)
    Alternative: Use pocketsphinx if Vosk not available
    """
    try:
        import speech_recognition as sr
        
        st.info("🗣️ Converting speech to text (using free Vosk model)...")
        
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        
        # Try Vosk first (completely free, offline)
        try:
            text = recognizer.recognize_vosk(audio)
            return text if text else "Could not recognize speech"
        except:
            # Fallback: Use PocketSphinx (also free)
            try:
                text = recognizer.recognize_sphinx(audio)
                return text if text else "Could not recognize speech"
            except:
                return "Note: Speech recognition needs Vosk. Please install: pip install vosk"
    
    except ImportError:
        st.warning("""
        ⚠️ Speech recognition libraries not installed.
        For FREE offline speech-to-text, install:
        ```
        pip install vosk pocketsphinx
        pip install SpeechRecognition
        ```
        """)
        return "Please install required packages for speech recognition"
    except Exception as e:
        return f"Error transcribing: {str(e)}"

def process_video_file_free(file_path):
    """Process uploaded video file using free methods"""
    try:
        from moviepy.editor import VideoFileClip
        
        st.info("🎙️ Extracting audio from uploaded video...")
        
        audio_path = os.path.join(tempfile.gettempdir(), 'temp_audio.wav')
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        video.close()
        
        text = transcribe_audio_vosk(audio_path)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return text
    except Exception as e:
        return f"Error: {str(e)}"

# ============ FREE TEXT PROCESSING ============

def preprocess_text(text):
    """Clean and normalize text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.!?,;\-]', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    return text.strip()

def extract_sentences(text):
    """Split text into sentences"""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    except:
        # Fallback to simple regex
        return re.split(r'[.!?]+', text)

# ============ FREE SUMMARIZATION (huggingface/transformers) ============

@st.cache_resource
def load_summarizer():
    """Load FREE summarization model"""
    try:
        from transformers import pipeline
        return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    except:
        st.warning("Installing summarization model (first time only)...")
        from transformers import pipeline
        return pipeline("summarization", model="facebook/bart-large-cnn", device=-1)

def generate_summary_free(text, num_points=5):
    """
    Generate bullet point summary using FREE BART model
    No API costs - runs locally
    """
    try:
        summarizer = load_summarizer()
        
        # Split into chunks
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1024:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        summaries = []
        progress_bar = st.progress(0)
        
        for idx, chunk in enumerate(chunks):
            if len(chunk.split()) < 50:
                continue
            try:
                summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
                progress_bar.progress(min((idx + 1) / len(chunks), 1.0))
            except:
                continue
        
        # Format as bullets
        bullets = []
        for summary in summaries:
            summary = summary.replace('\n', ' ')
            sents = re.split(r'[.!?]+', summary)
            
            for sent in sents:
                sent = sent.strip()
                if len(sent) > 10:
                    sent = sent[0].upper() + sent[1:] if sent else sent
                    bullets.append(f"• {sent}")
        
        return list(dict.fromkeys(bullets))[:num_points]
    
    except Exception as e:
        st.error(f"Summarization error: {str(e)}")
        return ["• Unable to generate summary"]

# ============ FREE QUIZ GENERATION ============

@st.cache_resource
def load_nlp():
    """Load FREE NLP model"""
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except:
        st.warning("Installing spaCy model (first time only)...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], capture_output=True)
        import spacy
        return spacy.load("en_core_web_sm")

def generate_quiz_free(text, num_questions=8, difficulty='mixed'):
    """
    Generate quiz questions using FREE spaCy NER
    No API costs - completely local
    """
    try:
        nlp = load_nlp()
        doc = nlp(text)
        
        # Extract entities as potential answers
        entities = [ent.text for ent in doc.ents]
        
        if not entities:
            return []
        
        quiz = []
        
        for i, entity in enumerate(entities[:num_questions]):
            # Generate question (simple template-based)
            questions_templates = [
                f"What is the significance of '{entity}' in this context?",
                f"Which of the following refers to '{entity}'?",
                f"'{entity}' is associated with which concept?",
                f"What does '{entity}' represent?",
                f"Which statement correctly describes '{entity}'?",
            ]
            
            question = random.choice(questions_templates)
            
            # Generate distractors
            distractors = random.sample(
                [e for e in entities if e != entity and len(e) > 2],
                min(3, len([e for e in entities if e != entity and len(e) > 2]))
            )
            
            if len(distractors) < 3:
                # Add default distractors if not enough
                distractors.extend(["Option A", "Option B", "Option C"][:3-len(distractors)])
            
            options = [entity] + distractors[:3]
            random.shuffle(options)
            correct_idx = options.index(entity)
            
            quiz.append({
                "id": i + 1,
                "question": question,
                "options": options,
                "correct_option": correct_idx,
                "difficulty": difficulty,
                "explanation": f"The correct answer is '{entity}'."
            })
        
        return quiz
    
    except Exception as e:
        st.error(f"Quiz generation error: {str(e)}")
        return []

# ============ FREE MIND MAP GENERATION ============

def generate_mindmap_free(text):
    """
    Generate mind map using FREE spaCy
    No API costs
    """
    try:
        nlp = load_nlp()
        doc = nlp(text)
        
        from collections import defaultdict
        concepts = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                concepts["👤 People"].append(ent.text)
            elif ent.label_ == "ORG":
                concepts["🏢 Organizations"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                concepts["📅 Timeline"].append(ent.text)
            elif ent.label_ in ["GPE", "LOC"]:
                concepts["📍 Locations"].append(ent.text)
            else:
                concepts["💡 Concepts"].append(ent.text)
        
        for key in concepts:
            concepts[key] = list(set(concepts[key]))
        
        mind_map = {
            "name": "Main Topic",
            "children": [
                {
                    "name": category,
                    "children": [{"name": item, "children": []} for item in items[:5]]
                }
                for category, items in concepts.items() if items
            ]
        }
        
        return mind_map
    
    except Exception as e:
        st.error(f"Mind map error: {str(e)}")
        return {"name": "Main Topic", "children": []}

# ============ MAIN UI ============

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3652/3652126.png", width=80)
with col2:
    st.title("🎓 AI Educational Assistant")
    st.markdown("*100% FREE - No API Costs - Runs Locally*")

st.markdown("""
<div class="warning-box">
<b>✅ What's FREE Here:</b>
- Video downloading (yt-dlp)
- Speech-to-text (Vosk - offline)
- Summarization (BART model)
- Quiz generation (spaCy NER)
- Mind maps (spaCy)
- Everything runs locally on your computer!
</div>
""", unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "📝 Summary", "❓ Quiz", "🧠 Mind Map"])

# TAB 1: INPUT
with tab1:
    st.header("Choose Your Input")
    
    input_method = st.radio(
        "Select input type:",
        ["📹 Video URL", "📤 Upload Video", "📄 Text Input"],
        horizontal=True
    )
    
    if input_method == "📹 Video URL":
        url = st.text_input("Enter video URL:", placeholder="https://www.youtube.com/...")
        st.caption("💰 FREE - Uses yt-dlp (no costs)")
        
        if url and st.button("🔗 Process URL", use_container_width=True, type="primary"):
            if validate_url(url):
                try:
                    with st.spinner("⏳ Processing video (may take 2-3 minutes)..."):
                        content = process_video_url_free(url)
                        if content and "Please install" not in content and "Error" not in content:
                            st.session_state.processed_data = {
                                'content': content,
                                'source': f"Video: {url[:50]}...",
                                'type': 'video'
                            }
                            st.success("✅ Video processed!")
                        else:
                            st.warning(f"⚠️ {content}")
                except Exception as e:
                    st.error(f"❌ {str(e)}")
            else:
                st.error("❌ Invalid URL")
    
    elif input_method == "📤 Upload Video":
        uploaded_file = st.file_uploader("Upload video:", type=['mp4', 'avi', 'mov', 'mkv'])
        st.caption("💰 FREE - Local processing only")
        
        if uploaded_file and st.button("⬆️ Process Video", use_container_width=True, type="primary"):
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                with st.spinner("⏳ Processing video (may take a minute)..."):
                    content = process_video_file_free(temp_path)
                    if content and "Please install" not in content:
                        st.session_state.processed_data = {
                            'content': content,
                            'source': uploaded_file.name,
                            'type': 'video'
                        }
                        st.success("✅ Video processed!")
                    else:
                        st.warning(f"⚠️ {content}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    else:  # Text
        text_area = st.text_area("Paste text:", height=200, placeholder="Enter educational content...")
        st.caption("💰 FREE - Text input processing")
        
        if text_area and st.button("📄 Process Text", use_container_width=True, type="primary"):
            if validate_text(text_area):
                processed = preprocess_text(text_area)
                st.session_state.processed_data = {
                    'content': processed,
                    'source': f"Text ({len(text_area)} chars)",
                    'type': 'text'
                }
                st.success("✅ Text processed!")
            else:
                st.error("❌ Text too short (min 50 words)")
    
    # Processing options
    if st.session_state.processed_data:
        st.divider()
        st.subheader("⚙️ Processing Options")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            summary_points = st.slider("Summary points:", 3, 10, 5)
        with col2:
            num_questions = st.slider("Quiz questions:", 3, 10, 6)
        with col3:
            difficulty = st.selectbox("Difficulty:", ["Easy", "Medium", "Hard", "Mixed"])
        
        include_mindmap = st.checkbox("Generate mind map", value=True)
        
        if st.button("🚀 Generate All (FREE)", use_container_width=True, type="primary"):
            progress_bar = st.progress(0)
            
            with st.spinner("📝 Generating summary..."):
                progress_bar.progress(25)
                summary = generate_summary_free(st.session_state.processed_data['content'], summary_points)
            
            with st.spinner("❓ Generating quiz..."):
                progress_bar.progress(50)
                quiz = generate_quiz_free(st.session_state.processed_data['content'], num_questions, difficulty.lower())
            
            mindmap = None
            if include_mindmap:
                with st.spinner("🧠 Generating mind map..."):
                    progress_bar.progress(75)
                    mindmap = generate_mindmap_free(st.session_state.processed_data['content'])
            
            progress_bar.progress(100)
            
            st.session_state.processed_data.update({
                'summary': summary,
                'quiz': quiz,
                'mindmap': mindmap
            })
            
            st.success("✅ All content generated (100% FREE)!")
            st.rerun()

# TAB 2: SUMMARY
with tab2:
    if st.session_state.processed_data and 'summary' in st.session_state.processed_data:
        data = st.session_state.processed_data
        
        st.header("📋 Summary")
        st.subheader(f"Source: {data['source']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Words", len(data['content'].split()))
        with col2:
            st.metric("✅ Points", len(data['summary']))
        with col3:
            st.metric("❓ Questions", len(data.get('quiz', [])))
        
        st.divider()
        st.subheader("🎯 Key Points")
        for i, point in enumerate(data['summary'], 1):
            st.markdown(f"{i}. {point}")
        
        summary_text = "\n".join(data['summary'])
        st.download_button("📥 Download Summary", data=summary_text, file_name="summary.txt", mime="text/plain")
        
        st.markdown("""
        <div class="warning-box">
        💡 <b>Tip:</b> Use these bullet points to study or teach!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👈 Process content first")

# TAB 3: QUIZ
with tab3:
    if st.session_state.processed_data and 'quiz' in st.session_state.processed_data:
        quiz_data = st.session_state.processed_data['quiz']
        
        if not quiz_data:
            st.warning("⚠️ Could not generate quiz questions. Try with more content.")
        else:
            st.header("❓ Interactive Quiz")
            score = 0
            total = len(quiz_data)
            
            for idx, question in enumerate(quiz_data, 1):
                st.subheader(f"Q{idx}. {question['question']}")
                
                selected = st.radio(
                    "Answer:",
                    options=question['options'],
                    key=f"q_{idx}",
                    horizontal=False
                )
                
                correct_idx = question.get('correct_option', 0)
                is_correct = selected == question['options'][correct_idx]
                
                if is_correct:
                    st.success(f"✅ Correct!")
                    score += 1
                else:
                    st.error(f"❌ Wrong! Correct: {question['options'][correct_idx]}")
                
                st.divider()
            
            st.subheader("📊 Results")
            percentage = (score / total * 100) if total > 0 else 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{score}/{total} ({percentage:.1f}%)")
            with col2:
                st.metric("Correct", score)
            with col3:
                st.metric("Incorrect", total - score)
            
            if percentage >= 80:
                st.success("🌟 Excellent! You've mastered the content!")
            elif percentage >= 60:
                st.info("👍 Good job! Review missed topics.")
            else:
                st.warning("📚 Keep studying!")
    else:
        st.info("👈 Generate quiz first")

# TAB 4: MIND MAP
with tab4:
    if st.session_state.processed_data and 'mindmap' in st.session_state.processed_data:
        mindmap = st.session_state.processed_data['mindmap']
        
        st.header("🧠 Mind Map")
        
        def display_mindmap(node, level=0):
            indent = "  " * level
            if level == 0:
                st.markdown(f"### 📌 {node.get('name', 'Root')}")
            else:
                st.markdown(f"{indent}**{node.get('name', 'Concept')}**")
            
            if 'children' in node and node['children']:
                for child in node['children']:
                    display_mindmap(child, level + 1)
        
        display_mindmap(mindmap)
        
        st.download_button(
            "📥 Download JSON",
            data=json.dumps(mindmap, indent=2),
            file_name="mindmap.json",
            mime="application/json"
        )
    else:
        st.info("👈 Generate mind map first")

# SIDEBAR
with st.sidebar:
    st.header("ℹ️ About (FREE Version)")
    st.markdown("""
    **AI Educational Assistant v2.0**
    
    ✅ 100% FREE
    ✅ No API costs
    ✅ Local processing
    ✅ Open source
    
    **Technologies:**
    - Streamlit
    - Transformers
    - spaCy
    - Vosk (speech-to-text)
    
    **No accounts needed!**
    """)
    
    st.divider()
    
    st.subheader("🆓 What's Free")
    st.markdown("""
    - Video download (yt-dlp)
    - Audio extraction (MoviePy)
    - Speech-to-text (Vosk)
    - Text summarization (BART)
    - Quiz generation (spaCy)
    - Mind maps (spaCy)
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared!")
    
    st.divider()
    
    st.subheader("📦 Install Offline Models")
    if st.button("Download Models (Run Once)"):
        st.info("Run these commands locally to pre-download models:")
        st.code("""
python -m spacy download en_core_web_sm
pip install vosk
        """)

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Made with ❤️ - 100% FREE & Open Source | No Hidden Charges</p>
</div>
""", unsafe_allow_html=True)
