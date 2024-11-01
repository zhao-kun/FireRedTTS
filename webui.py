# python promgram which provide a web interface to the user to interact with the system via gradio library

import gradio as gr
import os
import torchaudio
import argparse
import tempfile
from fireredtts.fireredtts import FireRedTTS
import numpy as np


sampling_rate = 24000

def tts_inference(text, prompt_wav='examples/prompt_1.wav', lang='zh'):
    # Model inference
    syn_audio = tts.synthesize(
        prompt_wav=prompt_wav,
        text=text,
        lang=lang,
    )[0].detach().cpu().numpy()

    # Normalize volume
    syn_audio = syn_audio / np.max(np.abs(syn_audio)) * 0.9
    
    # Convert audio data type
    syn_audio = (syn_audio * 32768).astype(np.int16)

    return sampling_rate, syn_audio

def main():
    iface = gr.Interface(
        fn=tts_inference,
        inputs=[
            gr.Textbox(label="Input text here"),
            gr.Audio(type="filepath", label="Upload reference audio"),
            gr.Dropdown(["en", "zh"], label="Select language"),
        ],
        outputs=gr.Audio(label="Generated audio"),
        title="FireRedTTS: A Foundation Text-To-Speech Framework for Industry-Level Generative Speech Applications")
    iface.launch(share=False, debug=True, server_port=args.port, server_name="0.0.0.0")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models',
                        help='local path')
    args = parser.parse_args()
    tts = FireRedTTS(
        config_path="configs/config_24k.json",
        pretrained_path='./pretrained_models',
    )
    main()
