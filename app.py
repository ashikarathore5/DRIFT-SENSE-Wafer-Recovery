import gradio as gr
import cv2
import numpy as np
import time
from inference import subpixel_parabolic_localization
from generate_dataset import create_semiconductor_wafer_with_mark, add_fab_artifacts

def run_live_drift_demo(drift_x, drift_y, noise_level):
    # 1. Base Wafer with Center Target
    gt_x, gt_y = int(drift_x), int(drift_y)
    base_wafer = create_semiconductor_wafer_with_mark(size=(1000, 1000), gt_x=gt_x, gt_y=gt_y)
    
    # 2. Add User-Controlled Noise
    img_float = base_wafer.astype(np.float32) / 255.0
    noise = np.random.normal(0, noise_level, img_float.shape)
    search_img = (np.clip(img_float + noise, 0, 1) * 255).astype(np.uint8)
    
    # 3. Reference Crop (100x100)
    crop_w, crop_h = 50, 50
    y1, y2 = gt_y - crop_h, gt_y + crop_h
    x1, x2 = gt_x - crop_w, gt_x + crop_w
    ref_img = cv2.resize(base_wafer[y1:y2, x1:x2], (100, 100))
    
    # 4. Run Sub-Pixel Inference Engine
    start = time.perf_counter()
    pred_x, pred_y, conf = subpixel_parabolic_localization(search_img, ref_img)
    latency_ms = (time.perf_counter() - start) * 1000
    
    # 5. Compute MAE
    pixel_error = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
    
    # 6. Draw Visual Bounding Box + Predicted Center Marker on Output Image
    output_vis = cv2.cvtColor(search_img, cv2.COLOR_GRAY2RGB)
    box_r = 50
    
    # Red Bounding Box around predicted region
    cv2.rectangle(output_vis, (int(pred_x) - box_r, int(pred_y) - box_r), 
                  (int(pred_x) + box_r, int(pred_y) + box_r), (255, 0, 0), 3)
    
    # Green Center Dot
    cv2.circle(output_vis, (int(pred_x), int(pred_y)), 6, (0, 255, 0), -1)
    
    metrics_text = f"""
    ### 📊 Real-Time Performance Analytics:
    - 🎯 **Ground Truth (GT):** ({gt_x}, {gt_y})
    - 📍 **Predicted Target:** ({pred_x:.3f}, {pred_y:.3f})
    - 📐 **Sub-Pixel Drift Error:** {pixel_error:.4f} pixels
    - ⚡ **Inference Latency:** {latency_ms:.2f} ms
    - 🔒 **Confidence Score:** {conf:.4f}
    """
    
    return output_vis, metrics_text

# --- Gradio UI Layout ---
with gr.Blocks(title="DRIFT-SENSE: Semiconductor Navigation Recovery", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔬 DRIFT-SENSE: AI Sub-Pixel Wafer Drift Recovery Engine")
    gr.Markdown("### Team: **Shift Happens** | Applied Materials Solution")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ Stage Control & Fab Artifacts")
            drift_x_slider = gr.Slider(200, 800, value=500, step=10, label="X-Axis Mechanical Drift (Pixels)")
            drift_y_slider = gr.Slider(200, 800, value=350, step=10, label="Y-Axis Mechanical Drift (Pixels)")
            noise_slider = gr.Slider(0.01, 0.25, value=0.08, step=0.01, label="Sensor Noise Intensity (Low SNR)")
            
            btn = gr.Button("🚀 Run Drift Recovery Engine", variant="primary")
            
        with gr.Column(scale=2):
            output_image = gr.Image(label="Live Wafer Inspection Area (Predicted Lock in RED)", type="numpy")
            metrics_display = gr.Markdown()

    btn.click(
        fn=run_live_drift_demo,
        inputs=[drift_x_slider, drift_y_slider, noise_slider],
        outputs=[output_image, metrics_display]
    )

if __name__ == "__main__":
    demo.launch()