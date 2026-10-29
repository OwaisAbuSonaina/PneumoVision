# frontend.py

# frontend.py

import os
import random
import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"

def get_risk_tier(probability: float):
    if probability >= 0.80:
        return {
            "tier": "High Probability",
            "desc": "Positive Findings Consistent with Pneumonia",
            "color": "#ef4444",
            "bg": "rgba(239, 68, 68, 0.12)"
        }
    elif probability > 0.53:
        return {
            "tier": "Moderate Probability",
            "desc": "Possible Infiltrates — Clinical Review Advised",
            "color": "#f97316",
            "bg": "rgba(249, 115, 22, 0.12)"
        }
    elif probability >= 0.35:
        return {
            "tier": "Borderline / Inconclusive",
            "desc": "Indeterminate Findings — Secondary Imaging Suggested",
            "color": "#eab308",
            "bg": "rgba(234, 179, 8, 0.12)"
        }
    else:
        return {
            "tier": "Low Probability",
            "desc": "No Acute Opacities Detected",
            "color": "#10b981",
            "bg": "rgba(16, 185, 129, 0.12)"
        }

def predict_pneumonia(image_path):
    if image_path is None:
        return "<div style='color: #ef4444; text-align: center; padding: 20px;'>Please upload an image first.</div>"
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            response = requests.post(API_URL, files=files)
        
        if response.status_code == 200:
            result = response.json()
            label = result.get("class", "Unknown")
            probability = float(result.get("confidence", 0.0))
            
            tier_info = get_risk_tier(probability)
            
            return f"""
            <div style='text-align: center; padding: 28px 20px; border-radius: 14px; background-color: {tier_info["bg"]}; border: 2px solid {tier_info["color"]}; height: 100%; display: flex; flex-direction: column; justify-content: center; box-sizing: border-box;'>
                <div style='display: inline-block; padding: 4px 12px; margin-bottom: 12px; border-radius: 9999px; background-color: {tier_info["color"]}; color: #ffffff; font-size: 0.82em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;'>
                    {tier_info["tier"]}
                </div>
                <h2 style='color: {tier_info["color"]}; margin: 0; font-size: 2.1em; letter-spacing: 0.5px;'>{label}</h2>
                <p style='margin: 10px 0 6px 0; font-size: 0.95em; color: var(--body-text-color-subdued);'>{tier_info["desc"]}</p>
                <div style='margin-top: 14px; font-size: 1.05em; font-weight: 600; color: var(--body-text-color);'>
                    Pneumonia Risk Index: <span style='color: {tier_info["color"]}; font-weight: 800;'>{probability:.1%}</span>
                </div>
            </div>
            """
        else:
            return f"<div style='color: #ef4444; padding: 20px;'>Error {response.status_code}: {response.text}</div>"
    except requests.exceptions.ConnectionError:
        return "<div style='color: #ef4444; padding: 20px;'>Error: Could not connect to API. Make sure FastAPI is running.</div>"
    except Exception as e:
        return f"<div style='color: #ef4444; padding: 20px;'>An error occurred: {str(e)}</div>"

def get_sample_images():
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        return []
    
    valid_exts = ('.png', '.jpg', '.jpeg')
    images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.lower().endswith(valid_exts)]
    
    if not images:
        return []
        
    sampled = random.sample(images, min(3, len(images)))
    return [[img] for img in sampled]

def refresh_dataset():
    return gr.Dataset(samples=get_sample_images())

custom_css = """
.center-container {
    max-width: 900px !important;
    margin: 40px auto !important;
    padding: 32px;
    background: var(--background-fill-primary);
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    border: 1px solid var(--border-color-primary);
}
.app-header {
    text-align: center;
    margin-bottom: 28px;
}
.app-header h1 {
    color: var(--primary-500);
    font-weight: 700;
    margin-bottom: 6px;
    font-size: 2.2em;
}
.app-header p {
    color: var(--body-text-color-subdued);
    font-size: 1.05em;
}
"""

theme = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="slate"
)

with gr.Blocks(css=custom_css, title="PneumoVision") as demo:
    with gr.Column(elem_classes="center-container"):
        
        gr.HTML("""
        <div class="app-header">
            <h1>🫁 PneumoVision AI</h1>
            <p>Upload a chest X-Ray to check for Pneumonia using our ResNet50 model</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Chest X-Ray Image", height=320)
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output_display = gr.HTML(
                    "<div style='text-align: center; padding: 120px 20px; color: var(--body-text-color-subdued); border: 2px dashed var(--border-color-primary); border-radius: 12px; height: 100%; box-sizing: border-box; display: flex; align-items: center; justify-content: center;'>Upload an image to see triage results.</div>"
                )
        
        gr.HTML("<hr style='margin: 30px 0; border: none; border-top: 1px solid var(--border-color-primary);' />")
        
        with gr.Row(equal_height=True):
            gr.HTML("<h3 style='margin: 0; padding-top: 5px; color: var(--body-text-color);'>💡 Try with Sample Images</h3>")
            shuffle_btn = gr.Button("🔄 Shuffle Suggestions", size="sm")
            
        sample_dataset = gr.Dataset(
            components=[image_input],
            samples=get_sample_images(),
            label="Click an image below to load it"
        )
        
    analyze_btn.click(
        fn=predict_pneumonia,
        inputs=image_input,
        outputs=output_display
    )
    
    sample_dataset.click(
        fn=lambda x: x[0],
        inputs=sample_dataset,
        outputs=image_input
    )
    
    shuffle_btn.click(
        fn=refresh_dataset,
        inputs=None,
        outputs=sample_dataset
    )

if __name__ == "__main__":
    demo.launch(theme=theme, inbrowser=True)