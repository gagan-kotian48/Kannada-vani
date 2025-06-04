import io
import streamlit as st
from pydub import AudioSegment
import numpy as np
import torch
import webrtcvad
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audiorecorder import audiorecorder

# ── 1) Load the fine-tuned Whisper model ──────────────────────────────
@st.cache_resource
def load_model():
    model_path = "./whisper-kn-finetuned-fleurs"
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.eval()
    return processor, model

processor, model = load_model()

# ── 2) Voice Activity Detection ────────────────────────────────────────
vad = webrtcvad.Vad(0)

def detect_speech_segments(audio_segment, frame_ms=30):
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

# ── 3) Streamlit UI ───────────────────────────────────────────────────
st.title("ಕನ್ನಡ ವಾಣಿ - Kannada Speech to Text")
st.write("Record audio, apply VAD, and transcribe in Kannada using your Whisper model.")

# Record audio from mic
audio = audiorecorder("🎤 Start Recording", "⏺️ Recording...")

audio_segment = None
if len(audio) > 0:
    st.audio(audio.export().read(), format="audio/wav")

    # 🔧 Export to proper WAV format using BytesIO
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    # Decode WAV safely
    audio_segment = AudioSegment.from_file(wav_io, format="wav")
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

# Show transcribe button only if audio is available
transcribe = st.button("📝 Transcribe", disabled=(audio_segment is None))

if transcribe and audio_segment is not None:
    with st.spinner("🔍 Detecting speech segments..."):
        speech_segments = detect_speech_segments(audio_segment, frame_ms=30)

    if not speech_segments:
        st.warning("⚠️ No speech detected.")
    else:
        # Split into 30-second chunks
        chunk_ms = 30_000
        chunks = []
        for seg in speech_segments:
            for i in range(0, len(seg), chunk_ms):
                chunks.append(seg[i : i + chunk_ms])
        st.success(f"✅ {len(chunks)} chunk(s) ready for transcription.")

        transcript_area = st.empty()
        progress_bar = st.progress(0)
        full_transcript = ""
        total = len(chunks)

        for idx, chunk in enumerate(chunks, start=1):
            st.info(f"🔤 Transcribing chunk {idx}/{total}...")
            samples = np.array(chunk.get_array_of_samples(), dtype=np.float32) / (1 << 15)
            inputs = processor(samples, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                pred_ids = model.generate(inputs.input_features)
            text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

            full_transcript += " " + text
            transcript_area.text_area("Transcription (so far)", full_transcript.strip(), height=200)
            progress_bar.progress(idx / total)

        st.success("🎉 Transcription complete!")
        st.subheader("📝 Final Transcript")
        st.text_area("Your Kannada text:", full_transcript.strip(), height=300)

        # Download button for transcript
        st.download_button("💾 Download Transcript", full_transcript.strip(), file_name="transcript.txt")
