import streamlit as st
import json
import random
import re
from collections import defaultdict

# Page configuration
st.set_page_config(
    page_title="AI Educational Assistant",
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
        padding: 15px 25px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    .success-box {background: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid #28a745;}
    .info-box {background: #d1ecf1; padding: 15px; border-radius: 8px; border-left: 4px solid #17a2b8;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3652/3652126.png", width=80)
with col2:
    st.title("🎓 AI Educational Assistant")
    st.markdown("*Summarize • Quiz • Learn*")

st.markdown('<div class="free-badge">☁️ 100% Cloud-Powered • No Installation • Completely FREE ☁️</div>', unsafe_allow_html=True)

st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📥 Input", "📝 Summary", "❓ Quiz", "🧠 Mind Map"])

# ============ TAB 1: INPUT ============
with tab1:
    st.header("📥 Input Your Content")
    
    input_method = st.radio(
        "Choose input type:",
        ["📄 Text Input", "📹 Video URL"],
        horizontal=True
    )
    
    if input_method == "📄 Text Input":
        st.subheader("Paste Educational Content")
        text_input = st.text_area(
            "Enter your text (minimum 50 words):",
            height=250,
            placeholder="Paste lecture notes, articles, research papers, or any educational content here..."
        )
        
        if st.button("✅ Process Text", use_container_width=True, type="primary"):
            if text_input:
                word_count = len(text_input.split())
                if word_count >= 50:
                    st.session_state.processed_data = {
                        'content': text_input,
                        'source': f"Text Input ({word_count} words)",
                        'type': 'text'
                    }
                    st.success(f"✅ Text processed! ({word_count} words)")
                else:
                    st.error(f"❌ Text too short. Need at least 50 words, you have {word_count}")
            else:
                st.error("❌ Please enter some text")
    
    else:  # Video URL
        st.subheader("Video URL Processing")
        st.info("""
        ℹ️ **Note:** Video processing requires local setup or API integration.
        
        For cloud deployment, please use **Text Input** instead.
        
        **Workaround:** Download video, use speech-to-text tool to convert to text, paste here!
        """)

# Processing options (show only if data processed)
if st.session_state.processed_data:
    st.divider()
    st.subheader("⚙️ Customization Options")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        summary_points = st.slider("Summary bullet points:", 3, 10, 5)
    with col2:
        num_questions = st.slider("Quiz questions:", 3, 8, 5)
    with col3:
        difficulty = st.selectbox("Question difficulty:", ["Easy", "Medium", "Hard", "Mixed"])
    
    include_mindmap = st.checkbox("Generate mind map", value=True)
    
    # Generate button
    if st.button("🚀 Generate Summary, Quiz & Mind Map", use_container_width=True, type="primary"):
        with st.spinner("🧠 Processing with AI models..."):
            try:
                content = st.session_state.processed_data['content']
                
                # ===== SUMMARIZATION =====
                st.info("📝 Generating summary...")
                
                # Load summarizer
                from transformers import pipeline
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
                
                # Split into chunks
                sentences = re.split(r'(?<=[.!?])\s+', content)
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 1024:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Generate summaries for each chunk
                summaries = []
                for chunk in chunks:
                    if len(chunk.split()) >= 50:
                        try:
                            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                            summaries.append(summary[0]['summary_text'])
                        except:
                            pass
                
                # Convert to bullet points
                bullets = []
                for summary in summaries:
                    summary = summary.replace('\n', ' ')
                    sents = re.split(r'[.!?]+', summary)
                    for sent in sents:
                        sent = sent.strip()
                        if len(sent) > 10:
                            sent = sent[0].upper() + sent[1:] if sent else sent
                            if sent not in bullets:
                                bullets.append(f"• {sent}")
                
                # ===== QUIZ GENERATION =====
                st.info("❓ Generating quiz questions...")
                
                import spacy
                nlp = spacy.load("en_core_web_sm")
                doc = nlp(content)
                
                # Extract entities
                entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
                entities = list(set(entities))  # Remove duplicates
                
                # Generate quiz
                quiz = []
                for i, entity in enumerate(entities[:num_questions]):
                    if not entity or len(entity) < 2:
                        continue
                    
                    # Question
                    questions_templates = [
                        f"What is the significance of '{entity}'?",
                        f"Which concept refers to '{entity}'?",
                        f"'{entity}' is associated with:",
                        f"What does '{entity}' mean?",
                        f"Which statement best describes '{entity}'?",
                    ]
                    question = random.choice(questions_templates)
                    
                    # Distractors
                    other_entities = [e for e in entities if e != entity]
                    distractors = random.sample(other_entities, min(3, len(other_entities)))
                    
                    # If not enough distractors, add placeholders
                    while len(distractors) < 3:
                        distractors.append(f"Option {len(distractors) + 1}")
                    
                    # Create options
                    options = [entity] + distractors
                    random.shuffle(options)
                    correct_idx = options.index(entity)
                    
                    quiz.append({
                        "id": i + 1,
                        "question": question,
                        "options": options,
                        "correct_option": correct_idx,
                        "difficulty": difficulty.lower(),
                        "explanation": f"The correct answer is: {entity}"
                    })
                
                # ===== MIND MAP GENERATION =====
                st.info("🧠 Creating mind map...")
                
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
                
                # Remove duplicates
                for key in concepts:
                    concepts[key] = list(set(concepts[key]))[:10]  # Limit to 10 per category
                
                mind_map = {
                    "name": "Main Topic",
                    "children": [
                        {
                            "name": category,
                            "children": [{"name": item, "children": []} for item in items]
                        }
                        for category, items in concepts.items() if items
                    ]
                }
                
                # Store all results
                st.session_state.processed_data.update({
                    'summary': bullets[:summary_points],
                    'quiz': quiz,
                    'mindmap': mind_map
                })
                
                st.success("✅ Generation complete!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Try with different content or fewer questions")

# ============ TAB 2: SUMMARY ============
with tab2:
    if st.session_state.processed_data and 'summary' in st.session_state.processed_data:
        data = st.session_state.processed_data
        
        st.header("📋 Summary & Insights")
        st.subheader(f"📌 Source: {data['source']}")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            word_count = len(data['content'].split())
            st.metric("📊 Content Words", f"{word_count:,}")
        with col2:
            st.metric("✅ Summary Points", len(data['summary']))
        with col3:
            st.metric("❓ Quiz Questions", len(data.get('quiz', [])))
        
        st.divider()
        
        # Summary display
        st.subheader("🎯 Key Points")
        for i, point in enumerate(data['summary'], 1):
            st.markdown(f"**{i}.** {point}")
        
        # Download button
        summary_text = "\n".join([f"{i}. {p}" for i, p in enumerate(data['summary'], 1)])
        st.download_button(
            label="📥 Download Summary (.txt)",
            data=summary_text,
            file_name="summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    else:
        st.info("👈 Go to **Input** tab and process content first")

# ============ TAB 3: QUIZ ============
with tab3:
    if st.session_state.processed_data and 'quiz' in st.session_state.processed_data:
        quiz_data = st.session_state.processed_data['quiz']
        
        if not quiz_data:
            st.warning("⚠️ Could not generate quiz. Please try with more content.")
        else:
            st.header("❓ Interactive Quiz")
            
            score = 0
            total = len(quiz_data)
            
            for idx, question in enumerate(quiz_data, 1):
                st.subheader(f"Question {idx}/{total}")
                st.write(f"**{question['question']}**")
                
                # Display difficulty
                difficulty_emoji = {
                    'easy': '🟢',
                    'medium': '🟡',
                    'hard': '🔴',
                    'mixed': '⚪'
                }
                st.caption(f"{difficulty_emoji.get(question.get('difficulty', 'medium'), '⚪')} {question.get('difficulty', 'medium').title()}")
                
                # Radio button for answer
                selected = st.radio(
                    "Select your answer:",
                    options=question['options'],
                    key=f"q_{idx}",
                    horizontal=False
                )
                
                # Check answer
                correct_idx = question.get('correct_option', 0)
                is_correct = selected == question['options'][correct_idx]
                
                if is_correct:
                    st.success(f"✅ **Correct!** {question.get('explanation', '')}")
                    score += 1
                else:
                    st.error(f"❌ **Incorrect!** Correct answer: {question['options'][correct_idx]}")
                    st.info(f"💡 {question.get('explanation', '')}")
                
                st.divider()
            
            # Results
            st.subheader("📊 Quiz Results")
            percentage = (score / total * 100) if total > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Score", f"{score}/{total}")
            with col2:
                st.metric("Percentage", f"{percentage:.1f}%")
            with col3:
                st.metric("Correct Answers", score)
            
            # Feedback
            st.divider()
            if percentage >= 80:
                st.success("🌟 **Excellent!** You've mastered this content!")
            elif percentage >= 60:
                st.info("👍 **Good job!** Review the missed questions.")
            else:
                st.warning("📚 **Keep studying!** Try again after reviewing the content.")
    else:
        st.info("👈 Go to **Input** tab and process content first")

# ============ TAB 4: MIND MAP ============
with tab4:
    if st.session_state.processed_data and 'mindmap' in st.session_state.processed_data:
        mindmap = st.session_state.processed_data['mindmap']
        
        st.header("🧠 Concept Mind Map")
        
        def display_mindmap(node, level=0):
            """Recursively display mind map"""
            indent = "  " * level
            if level == 0:
                st.markdown(f"### 📌 {node.get('name', 'Root')}")
            else:
                st.markdown(f"{indent}**{node.get('name', 'Concept')}**")
            
            if 'children' in node and node['children']:
                for child in node['children']:
                    display_mindmap(child, level + 1)
        
        # Display mind map
        display_mindmap(mindmap)
        
        st.divider()
        
        # Download JSON
        st.download_button(
            label="📥 Download Mind Map (JSON)",
            data=json.dumps(mindmap, indent=2),
            file_name="mindmap.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("👈 Go to **Input** tab and process content first")

# ============ SIDEBAR ============
with st.sidebar:
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ### AI Educational Assistant v2.0
    
    **Features:**
    - 📝 AI-powered summarization
    - ❓ Auto-generated quizzes
    - 🧠 Interactive mind maps
    
    **Technology:**
    - Streamlit (UI)
    - Transformers (BART, spaCy)
    - Runs on cloud servers
    
    **Cost:** ✅ Completely FREE
    """)
    
    st.divider()
    
    st.subheader("🚀 How to Use")
    st.markdown("""
    1. Go to **Input** tab
    2. Paste educational text
    3. Click "Generate"
    4. View summary, quiz, mind map
    5. Download results
    """)
    
    st.divider()
    
    st.subheader("💡 Tips")
    st.markdown("""
    - **Minimum:** 50 words
    - **Best:** 500-5000 words
    - **Formats:** Lectures, articles, research
    - **Time:** 1-2 minutes per 1000 words
    """)
    
    st.divider()
    
    st.subheader("☁️ Cloud Powered")
    st.markdown("""
    ✅ No installation
    ✅ No downloads
    ✅ No API keys
    ✅ 100% FREE
    
    Made with ❤️ for learners
    """)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🎓 AI Educational Assistant | ☁️ Powered by Streamlit Cloud | 100% FREE & Open Source</p>
    <p style="font-size: 12px;">No data is stored. All processing is temporary.</p>
</div>
""", unsafe_allow_html=True)
