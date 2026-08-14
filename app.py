from flask import Flask, render_template_string, request, jsonify
import cv2
import numpy as np
import base64
from localize import match_drift

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>DRIFT-SENSE Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; background: #0f172a; color: #f8fafc; text-align: center; padding: 20px; }
        .card { background: #1e293b; padding: 20px; border-radius: 12px; display: inline-block; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
        h1 { color: #38bdf8; }
        .btn { background: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #0369a1; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🚀 DRIFT-SENSE Wafer Alignment System</h1>
        <p>Phase-Correlation Precision Wafer Drift Localizer</p>
        <p>Status: <strong style="color: #4ade80;">Active & Ready</strong></p>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)