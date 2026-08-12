import os
import cv2
import time
import numpy as np
import gradio as gr
from inference import predict_center
from generate_dataset import generate_pair, _init_module_level_grids

# Initialize dataset grids for real-time generation
_init_module_level_grids()

def run_live_drift_demo(noise_level):
    # 1. Generate realistic SEM pair using our physics engine
    rng = np.random.default_rng()
    search_img, ref_img, gt_center, pattern, landmark = generate_pair(rng)
    gt_x, gt_y = gt_center

    # 2. Save temporary images for inference engine
    os.makedirs("temp_demo", exist_ok=True)
    search_path = "temp_demo/search.png"
    ref_path = "temp_demo/ref.png"
    cv2.imwrite(search_path, search_img)
    cv2.imwrite(ref_path, ref_img)

    # 3. Run Sub-Pixel Inference Engine
    start = time.perf_counter()
    pred_x, pred_y, conf = predict_center(search_path, ref_path)
    latency_ms = (time.perf_counter() - start) * 1000.0

    # 4. Compute Sub-Pixel Error
    pixel_error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)

    # 5. Draw Visual Bounding Box & Predicted Center
    output_vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
    box_r = 50

    # Red Bounding Box around predicted region
    cv2.rectangle(output_vis, 
                  (int(pred_x) - box_r, int(pred_y) - box_r), 
                  (int(pred_x) + box_r, int(pred_y) + box_r), 
                  (255, 0, 0), 3)

    # Green Center Dot for Prediction
    cv2.circle(output_vis, (int(pred_x), int(pred_y)), 6, (0, 255, 0), -1)
    
    # Blue Dot for Ground Truth Center
    cv2.circle(output_vis, (int(gt_x), int(gt_y)), 4, (0, 0, 255), -1)

    metrics_text = f"""
    ### 📊 Real-Time Performance Analytics:
    - 📐 **Pattern Geometry:** `{pattern.upper()}` | **Landmark:** `{landmark}`
    - 🎯 **Ground Truth (GT Center):** ({gt_x:.2f}, {gt_y:.2f})
    - 📍 **Predicted Lock:** ({pred_x:.4f}, {pred_y:.4f})
    - 🔬 **Sub-Pixel Drift Error:** **{pixel_error:.4f} pixels**
    - ⚡ **Inference Latency:** **{latency_ms:.2f} ms**
    - 🔒 **Match Confidence:** **{conf:.4f}**
    """

    return output_vis, metrics_text

# --- Gradio UI Layout ---
with gr.Blocks(title="DRIFT-SENSE: Semiconductor Alignment Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 DRIFT-SENSE: Sub-Pixel Wafer Drift Recovery Engine")
    gr.Markdown("### Applied Materials Semiconductor Metrology Benchmark Demo")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Real-Time SEM Simulation")
            noise_slider = gr.Slider(0.01, 0.25, value=0.08, step=0.01, label="SEM Noise & Charging Intensity")
            
            btn = gr.Button("🚀 Simulate Drift Recovery", variant="primary")
            
        with gr.Column(scale=2):
            output_image = gr.Image(label="Live SEM Inspection Area (Predicted Lock in RED)", type="numpy")
            metrics_display = gr.Markdown()

    btn.click(
        fn=run_live_drift_demo,
        inputs=[noise_slider],
        outputs=[output_image, metrics_display]
    )

if __name__ == "__main__":
    demo.launch()
    