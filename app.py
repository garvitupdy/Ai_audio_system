import streamlit as st
from dotenv import load_dotenv
from audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from database.db_manager import TranscriptDatabase
import json
from datetime import datetime

load_dotenv()


st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #4CAF50;
        --background-color: #0E1117;
        --card-background: #1E1E1E;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 100%);
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #6C63FF 0%, #5a52d5 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.3);
        text-align: center;
    }
    
    .custom-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .custom-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2a2a3e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #6C63FF;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(108, 99, 255, 0.4);
    }
    
    /* Success card */
    .success-card {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        border-left-color: #4CAF50;
    }
    
    /* Warning card */
    .warning-card {
        background: linear-gradient(135deg, #4a3c1a 0%, #5a4d2d 100%);
        border-left-color: #FFC107;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF 0%, #5a52d5 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(108, 99, 255, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a52d5 0%, #4a42c5 100%);
        box-shadow: 0 6px 20px rgba(108, 99, 255, 0.5);
        transform: translateY(-2px);
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #2a2a3e;
        color: white;
        border: 2px solid #3a3a4e;
        border-radius: 8px;
        padding: 0.75rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #6C63FF;
        box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.2);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #2a2a3e;
        border-radius: 8px;
        font-weight: 600;
        color: white;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: fadeIn 0.5s ease;
    }
    
    .user-message {
        background: linear-gradient(135deg, #6C63FF 0%, #5a52d5 100%);
        margin-left: 20%;
        color: white;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #2a2a3e 0%, #3a3a4e 100%);
        margin-right: 20%;
        color: white;
        border-left: 4px solid #4CAF50;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* History item */
    .history-item {
        background: #2a2a3e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 3px solid transparent;
    }
    
    .history-item:hover {
        background: #3a3a4e;
        border-left-color: #6C63FF;
        transform: translateX(5px);
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #6C63FF;
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #3a3a4e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-top: 3px solid #6C63FF;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #6C63FF;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.7);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2a2a3e;
        border-radius: 8px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6C63FF;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_database():
    return TranscriptDatabase()

db = init_database()


if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_transcript_id' not in st.session_state:
    st.session_state.current_transcript_id = None

st.markdown("""
<div class="custom-header">
    <h1>🎥 AI Video Assistant</h1>
    <p>Transform your videos into actionable insights with AI</p>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📚 Transcript History")
    st.markdown("---")
    
    transcripts = db.get_all_transcripts()
    
    if transcripts:
        for t_id, title, source, language, created_at in transcripts:
            with st.container():
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"📄 {title[:30]}...", key=f"hist_{t_id}", use_container_width=True):
                        st.session_state.current_transcript_id = t_id
                        st.session_state.current_result = db.get_transcript(t_id)
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{t_id}"):
                        db.delete_transcript(t_id)
                        st.rerun()
                
                st.caption(f"🕒 {created_at}")
                st.markdown("---")
    else:
        st.info("No transcripts yet. Create your first one!")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This AI-powered tool helps you:
    - 📝 Transcribe videos
    - 📊 Generate summaries
    - ✅ Extract action items
    - 🔑 Identify key decisions
    - ❓ Find open questions
    - 💬 Chat with your content
    """)


tab1, tab2 = st.tabs(["🎬 New Transcript", "📖 View & Analyze"])

with tab1:
    st.markdown("### 🎯 Create New Transcript")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        source = st.text_input(
            "📍 Enter YouTube URL or Local File Path",
            placeholder="https://youtube.com/watch?v=... or /path/to/video.mp4",
            help="Provide a YouTube URL or path to a local video file"
        )
    with col2:
        language = st.selectbox(
            "🌐 Language",
            options=["english", "hinglish"],
            index=0
        )
    
    if st.button("🚀 Start Processing", type="primary", use_container_width=True):
        if source:
            try:
                with st.spinner("🔄 Processing your video..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                   
                    status_text.text("📥 Extracting audio from video...")
                    progress_bar.progress(15)
                    chunks = process_input(source)
                    
                    
                    status_text.text("🎤 Transcribing audio (this may take a while)...")
                    progress_bar.progress(35)
                    transcript = transcribe_all(chunks, language)
                    
                    status_text.text("✨ Generating title...")
                    progress_bar.progress(55)
                    title = generate_title(transcript)
                    

                    status_text.text("📝 Creating summary...")
                    progress_bar.progress(75)
                    summary = summarize(transcript)
                    
             
                    status_text.text("💾 Saving to database...")
                    progress_bar.progress(85)
                    
                    temp_data = {
                        'title': title,
                        'source': source,
                        'language': language,
                        'transcript': transcript,
                        'summary': summary,
                        'action_items': '',
                        'key_decisions': '',
                        'open_questions': ''
                    }
                    
                    
                    transcript_id = db.add_transcript(temp_data)
                    st.session_state.current_transcript_id = transcript_id
                    
                    
                    status_text.text("🔧 Building RAG chain and vector store...")
                    progress_bar.progress(95)
                    rag_chain = build_rag_chain(transcript, transcript_id)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    
                    st.session_state.current_result = {
                        'id': transcript_id,
                        'title': title,
                        'source': source,
                        'language': language,
                        'transcript': transcript,
                        'summary': summary,
                        'action_items': None,
                        'key_decisions': None,
                        'open_questions': None,
                        'rag_chain': rag_chain
                    }
                    
                    st.session_state.chat_history = []  
                    st.session_state.processing_complete = True
                    
                    st.success("✅ Video processed successfully! Go to 'View & Analyze' tab to see results.")
                    st.balloons()
                    
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
        else:
            st.warning("⚠️ Please provide a video source!")
    
    
    st.markdown("---")
    st.markdown("""
    <div class="info-card">
        <h3>💡 Tips for Best Results</h3>
        <ul>
            <li><strong>YouTube URLs:</strong> Use full video URLs (not shorts or playlists)</li>
            <li><strong>Local Files:</strong> Supports MP4, MP3, WAV, and other common formats</li>
            <li><strong>Language:</strong> Choose 'hinglish' for mixed Hindi-English content</li>
            <li><strong>Processing Time:</strong> Varies based on video length (typically 1-5 minutes)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    
    with st.expander("📌 Example URLs"):
        st.code("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        st.code("/Users/username/Documents/meeting_recording.mp4")
        st.code("C:\\Users\\username\\Videos\\presentation.mp4")


with tab2:
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        
        st.markdown(f"""
        <div class="info-card">
            <h2>📌 {result['title']}</h2>
        </div>
        """, unsafe_allow_html=True)
        
       
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            word_count = len(result['transcript'].split())
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{word_count:,}</div>
                <div class="metric-label">Words Transcribed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{result['language'].title()}</div>
                <div class="metric-label">Language</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            chars = len(result['transcript'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{chars:,}</div>
                <div class="metric-label">Characters</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            
            reading_time = max(1, word_count // 200)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{reading_time}</div>
                <div class="metric-label">Min Read Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        
        with st.expander("📋 Summary", expanded=True):
            st.markdown(f"""
            <div class="success-card">
                {result['summary']}
            </div>
            """, unsafe_allow_html=True)
        
    
        with st.expander("📄 Full Transcript"):
            st.text_area("", result['transcript'], height=400, disabled=True, label_visibility="collapsed")
            
           
            st.download_button(
                label="📥 Download Transcript",
                data=result['transcript'],
                file_name=f"{result['title']}_transcript.txt",
                mime="text/plain"
            )
        
        st.markdown("---")
        st.markdown("### 🔍 Advanced Analysis")
        st.markdown("Click the buttons below to extract specific insights from your transcript")
        

        col1, col2, col3 = st.columns(3)
        
       
        with col1:
            if st.button("✅ Extract Action Items", use_container_width=True, key="btn_action_items"):
                with st.spinner("🔍 Extracting action items..."):
                    try:
                        action_items = extract_action_items(result['transcript'])
                        st.session_state.current_result['action_items'] = action_items
                        
                        # Update database
                        if st.session_state.current_transcript_id:
                            db.update_analysis(
                                st.session_state.current_transcript_id,
                                'action_items',
                                json.dumps(action_items) if isinstance(action_items, list) else action_items
                            )
                        st.success("✅ Action items extracted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting action items: {str(e)}")
        
   
        with col2:
            if st.button("🔑 Extract Key Decisions", use_container_width=True, key="btn_key_decisions"):
                with st.spinner("🔍 Extracting key decisions..."):
                    try:
                        decisions = extract_key_decisions(result['transcript'])
                        st.session_state.current_result['key_decisions'] = decisions
                        
                       
                        if st.session_state.current_transcript_id:
                            db.update_analysis(
                                st.session_state.current_transcript_id,
                                'key_decisions',
                                json.dumps(decisions) if isinstance(decisions, list) else decisions
                            )
                        st.success("✅ Key decisions extracted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting key decisions: {str(e)}")
        
        
        with col3:
            if st.button("❓ Extract Questions", use_container_width=True, key="btn_questions"):
                with st.spinner("🔍 Extracting open questions..."):
                    try:
                        questions = extract_questions(result['transcript'])
                        st.session_state.current_result['open_questions'] = questions
                        
                       
                        if st.session_state.current_transcript_id:
                            db.update_analysis(
                                st.session_state.current_transcript_id,
                                'open_questions',
                                json.dumps(questions) if isinstance(questions, list) else questions
                            )
                        st.success("✅ Open questions extracted!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting questions: {str(e)}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        

        analysis_cols = st.columns(3)
        
        
        with analysis_cols[0]:
            if result.get('action_items') and result['action_items']:
                with st.expander("✅ Action Items", expanded=True):
                    st.markdown(f"""
                    <div class="info-card">
                        {result['action_items']}
                    </div>
                    """, unsafe_allow_html=True)
     
        with analysis_cols[1]:
            if result.get('key_decisions') and result['key_decisions']:
                with st.expander("🔑 Key Decisions", expanded=True):
                    st.markdown(f"""
                    <div class="info-card">
                        {result['key_decisions']}
                    </div>
                    """, unsafe_allow_html=True)
        
        
        with analysis_cols[2]:
            if result.get('open_questions') and result['open_questions']:
                with st.expander("❓ Open Questions", expanded=True):
                    st.markdown(f"""
                    <div class="info-card">
                        {result['open_questions']}
                    </div>
                    """, unsafe_allow_html=True)
        
       
        st.markdown("---")
        st.markdown("### 💬 Chat with Your Transcript")
        st.markdown("Ask questions about the content and get AI-powered answers")
        
        
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if message['role'] == 'user':
                        st.markdown(f"""
                        <div class="chat-message user-message">
                            <strong>👤 You:</strong><br>{message['content']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant:</strong><br>{message['content']}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("💡 Start a conversation by asking a question below!")
        
        
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "Ask a question",
                key="chat_input",
                placeholder="e.g., What were the main topics discussed?",
                label_visibility="collapsed"
            )
        with col2:
            send_button = st.button("Send 📤", use_container_width=True, type="primary")
        
       
        with st.expander("💡 Suggested Questions"):
            st.markdown("""
            - What are the key takeaways from this transcript?
            - Who were the main speakers or participants?
            - What decisions were made during the discussion?
            - Are there any deadlines mentioned?
            - What are the next steps?
            """)
        
        if send_button and question:
            
            st.session_state.chat_history.append({
                'role': 'user',
                'content': question
            })
            
            with st.spinner("🤔 Thinking..."):
                try:
                    if 'rag_chain' in result and result['rag_chain']:
                        answer = ask_question(result['rag_chain'], question)
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': answer
                        })
                    else:
                        st.error("RAG chain not available. Please reload the transcript.")
                except Exception as e:
                    st.error(f"Error getting answer: {str(e)}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': f"Sorry, I encountered an error: {str(e)}"
                    })
            
            st.rerun()
        

        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Chat History", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with col2:
            if st.session_state.chat_history:
               
                chat_export = "\n\n".join([
                    f"{'You' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in st.session_state.chat_history
                ])
                st.download_button(
                    label="📥 Export Chat",
                    data=chat_export,
                    file_name=f"{result['title']}_chat.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
    else:
       
        st.markdown("""
        <div class="info-card">
            <h3>👋 Welcome to AI Video Assistant!</h3>
            <p>Get started by:</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="success-card">
                <h4>🎬 Option 1: Create New Transcript</h4>
                <ol>
                    <li>Go to the <strong>"New Transcript"</strong> tab</li>
                    <li>Enter a YouTube URL or file path</li>
                    <li>Select the language</li>
                    <li>Click <strong>"Start Processing"</strong></li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-card">
                <h4>📚 Option 2: Load from History</h4>
                <ol>
                    <li>Check the <strong>sidebar</strong> on the left</li>
                    <li>Click on any saved transcript</li>
                    <li>It will load here instantly</li>
                    <li>Start analyzing or chatting!</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
     
        st.markdown("""
        <div class="info-card">
            <h3>✨ What You Can Do</h3>
        </div>
        """, unsafe_allow_html=True)
        
        feature_cols = st.columns(3)
        
        with feature_cols[0]:
            st.markdown("""
            **📊 Smart Analysis**
            - Auto-generated summaries
            - Action item extraction
            - Key decision points
            - Open questions tracking
            """)
        
        with feature_cols[1]:
            st.markdown("""
            **💬 AI Chat**
            - Ask anything about content
            - Context-aware responses
            - Chat history tracking
            - Export conversations
            """)
        
        with feature_cols[2]:
            st.markdown("""
            **💾 Smart Storage**
            - Automatic saving
            - Quick access history
            - Vector search enabled
            - Persistent RAG chains
            """)