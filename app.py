import gradio as gr
from utils import send_message

gemini = gr.ChatInterface(
    fn=send_message,
    title="GeminiChat",
    multimodal=True,
)

gemini.launch()