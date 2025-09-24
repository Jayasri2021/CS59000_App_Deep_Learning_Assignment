import gradio as gr
from transformers import pipeline

# Load Whisper Small pipeline
asr = pipeline("automatic-speech-recognition", model="openai/whisper-small")

def transcribe(audio_file):
    if audio_file is None:
        return "Please upload an audio file."
    # Gradio passes file path when type="filepath"
    result = asr(audio_file)
    return result["text"]

with gr.Blocks() as demo:
    gr.Markdown("# 🎙 Whisper Small Speech-to-Text App")
    gr.Markdown("Upload an audio file (.wav, .m4a, .mp3) and get the transcript.")
    
    audio_in = gr.Audio(type="filepath", label="Upload audio")
    text_out = gr.Textbox(label="Transcript", lines=6)
    
    audio_in.change(fn=transcribe, inputs=audio_in, outputs=text_out)

if __name__ == "__main__":
    demo.launch()
