# streamlit_app.py

import io
import time
import streamlit as st
from pydub import AudioSegment
import numpy as np
import torch
import webrtcvad
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="ಕನ್ನಡ ವಾಣಿ - Kannada Speech to Text",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS Styling ──────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styling */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Top logo styling */
    .top-logo {
        position: absolute;
        top: 20px;
        left: 20px;
        z-index: 1000;
    }
    
    .top-logo img {
        width: 60px;
        height: 60px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .top-logo img:hover {
        transform: scale(1.1);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }

    /* Main container styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .logo-text {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b35, #f7931e, #ffd23f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .tagline {
        color: white;
        font-size: 1.4rem;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .main-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Upload section styling */
    .upload-section {
        border: 3px dashed rgba(255,255,255,0.3);
        border-radius: 15px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(255,255,255,0.05);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #ff6b35;
        background: rgba(255,107,53,0.1);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #ff6b35, #f7931e) !important;
        color: white !important;
        border: none !important;
        border-radius: 50px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        box-shadow: 0 10px 20px rgba(255,107,53,0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 15px 30px rgba(255,107,53,0.4) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b35, #f7931e, #ffd23f) !important;
        border-radius: 10px !important;
    }
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffd23f;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.95) !important;
        border-radius: 15px !important;
        border: none !important;
        font-size: 1.1rem !important;
        line-height: 1.8 !important;
        font-family: 'Noto Sans Kannada', 'Poppins', sans-serif !important;
        box-shadow: inset 0 5px 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Audio player styling */
    .stAudio {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }
    
    /* Status indicators */
    .status-success {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .status-processing {
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        text-align: center;
        margin: 1rem 0;
        font-weight: 500;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Live transcript styling */
    .live-transcript {
        border-left: 4px solid #ff6b35;
        background: rgba(255,107,53,0.05);
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .logo-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .logo-icon {
        width: 80px;
        height: 80px;
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        box-shadow: 0 10px 30px rgba(255,107,53,0.3);
        animation: pulse 3s ease-in-out infinite;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the fine-tuned Whisper model"""
    try:
        model_path = "./whisper-kn-finetuned-fleurs"
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model.eval()
        return processor, model, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, False

# ── Voice Activity Detection ────────────────────────────────────────────────
@st.cache_resource
def initialize_vad():
    """Initialize Voice Activity Detection"""
    return webrtcvad.Vad(0)

def detect_speech_segments(audio_segment, frame_ms=30):
    """Detect speech segments in audio"""
    vad = initialize_vad()
    seg16 = (
        audio_segment
        .set_frame_rate(16000)
        .set_channels(1)
        .set_sample_width(2)
    )
    pcm = seg16.raw_data
    step = int(16000 * frame_ms / 1000) * 2

    flags = [
        (i, vad.is_speech(pcm[i:i+step], sample_rate=16000))
        for i in range(0, len(pcm), step)
        if len(pcm[i:i+step]) == step
    ]

    segments, start = [], None
    for i, is_speech in flags:
        if is_speech and start is None:
            start = i
        elif not is_speech and start is not None:
            segments.append(seg16[start // 2 : i // 2])
            start = None
    if start is not None:
        segments.append(seg16[start // 2 :])
    return segments

# ── Audio Visualization ─────────────────────────────────────────────────────
def create_waveform_plot(audio_segment):
    """Create a waveform visualization"""
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2)).mean(axis=1)
    
    time_axis = np.linspace(0, len(samples) / audio_segment.frame_rate, len(samples))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=samples,
        mode='lines',
        name='Waveform',
        line=dict(color='#ff6b35', width=1)
    ))
    
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300
    )
    
    return fig

# ── Main Application ────────────────────────────────────────────────────────
def main():
    inject_custom_css()
    
    # Top left logo using Streamlit's method
    col1, col2, col3 = st.columns([1, 8, 1])
    with col1:
        try:
            st.image("image.png", width=180)
        except:
            # Fallback if image not found
            st.markdown("""
            <div style="width: 180px; height: 180px; background: linear-gradient(45deg, #ff6b35, #f7931e); 
                        border-radius: 25px; display: flex; align-items: center; justify-content: center; 
                        font-size: 4.5rem; color: white; font-weight: bold;">K</div>
            """, unsafe_allow_html=True)
    
    # Header with Logo
    st.markdown("""
    <div class="main-header">
        <div class="logo-container">
            <div class="logo-icon">🎤</div>
            <div class="logo-text">ಕನ್ನಡ ವಾಣಿ</div>
        </div>
        <div class="tagline">Transform your Kannada speech into text with AI precision</div>
        <div class="subtitle">Supports WAV, MP3, FLAC, OGG</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'transcription_history' not in st.session_state:
        st.session_state.transcription_history = []
    if 'current_transcript' not in st.session_state:
        st.session_state.current_transcript = ""
    
    # Load model
    processor, model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Failed to load the Whisper model. Please check the model path.")
        return
    
    # Main container
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        
        # File upload section
        st.markdown("### 🎤 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg"],
            help="Upload your Kannada audio file for transcription"
        )
        
        if uploaded_file is not None:
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{uploaded_file.size // 1024}</div>
                    <div class="metric-label">KB</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">{uploaded_file.type.split('/')[-1].upper()}</div>
                    <div class="metric-label">Format</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-number">✓</div>
                    <div class="metric-label">Ready</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Audio player
            st.markdown("### 🎵 Audio Preview")
            st.audio(uploaded_file, format=uploaded_file.type)
            
            # Process audio file
            audio_bytes = uploaded_file.read()
            ext = uploaded_file.name.rsplit(".", 1)[-1]
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=ext)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Show waveform
            with st.expander("📊 Audio Waveform Analysis", expanded=False):
                fig = create_waveform_plot(audio)
                st.plotly_chart(fig, use_container_width=True)
                
                # Audio stats
                duration = len(audio) / 1000
                st.markdown(f"""
                **Audio Statistics:**
                - Duration: {duration:.1f} seconds
                - Sample Rate: {audio.frame_rate} Hz
                - Channels: {audio.channels}
                - Sample Width: {audio.sample_width} bytes
                """)
            
            # Transcription section
            st.markdown("### 🚀 Start Transcription")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                transcribe_btn = st.button("🎯 Begin Transcription", use_container_width=True)
            with col2:
                show_live_updates = st.checkbox("Live Updates", value=True)
            
            if transcribe_btn:
                start_time = time.time()
                
                # VAD segmentation
                with st.spinner("🔍 Analyzing audio and detecting speech segments..."):
                    speech_segments = detect_speech_segments(audio, frame_ms=30)
                
                if not speech_segments:
                    st.warning("⚠️ No speech detected in the uploaded file. Please try another file.")
                    return
                
                # Prepare chunks
                chunk_ms = 30_000
                chunks = []
                for seg in speech_segments:
                    for i in range(0, len(seg), chunk_ms):
                        chunks.append(seg[i : i + chunk_ms])
                
                total_chunks = len(chunks)
                st.success(f"✅ Found {len(speech_segments)} speech segments")
                
                # Progress tracking
                progress_container = st.container()
                transcript_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                # Live transcript display
                if show_live_updates:
                    with transcript_container:
                        st.markdown("### 📝 Live Transcription")
                        live_transcript_area = st.empty()
                
                # Process chunks
                full_transcript = ""
                word_count = 0
                
                for idx, chunk in enumerate(chunks, start=1):
                    # Update progress
                    progress = idx / total_chunks
                    progress_bar.progress(progress)
                    status_text.markdown(f"""
                    <div class="status-processing">
                        🔄 Converting speech to text...
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Transcribe chunk
                    samples = np.array(chunk.get_array_of_samples(), dtype=np.float32) / (1 << 15)
                    inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    with torch.no_grad():
                        pred_ids = model.generate(inputs.input_features)
                    
                    text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
                    
                    if text.strip():
                        full_transcript += " " + text.strip()
                        word_count += len(text.split())
                    
                    # Update live transcript
                    if show_live_updates and full_transcript.strip():
                        with live_transcript_area:
                            st.text_area(
                                "Live Transcription:",
                                value=full_transcript.strip(),
                                height=200,
                                key=f"live_transcript_{idx}",
                                help="Transcription in progress..."
                            )
                    
                    # Small delay for visual effect
                    time.sleep(0.5)
                
                # Completion
                processing_time = time.time() - start_time
                
                status_text.markdown("""
                <div class="status-success">
                    ✅ Transcription completed successfully!
                </div>
                """, unsafe_allow_html=True)
                
                # Store results
                st.session_state.current_transcript = full_transcript.strip()
                
                # Final results
                st.markdown("### 📄 Final Transcription Results")
                
                # Stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{word_count}</div>
                        <div class="metric-label">Words Found</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{processing_time:.1f}s</div>
                        <div class="metric-label">Processing Time</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    words_per_minute = (word_count / processing_time) * 60 if processing_time > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-number">{words_per_minute:.0f}</div>
                        <div class="metric-label">Words/Min</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Transcript display and actions
                st.markdown("### 📝 Your Kannada Transcript")
                transcript_text = st.text_area(
                    "Transcribed Text:",
                    value=full_transcript.strip(),
                    height=300,
                    help="You can edit the transcription if needed"
                )
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download Transcript",
                        data=transcript_text,
                        file_name=f"kannada_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    if st.button("🔄 Clear All", use_container_width=True):
                        st.session_state.current_transcript = ""
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()