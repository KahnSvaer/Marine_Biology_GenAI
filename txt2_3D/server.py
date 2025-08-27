from flask import Flask, request, jsonify
import os
import torch
from generate_3d import generate_3d  # this should expect models passed in

# -------------------------------
# Load models globally at startup
# -------------------------------
print("[INFO] Loading models once at startup...")

# if your generate_3d internally loads models, move that logic here
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

DEVICE = torch.device(
    "cuda:1" if torch.cuda.device_count() > 1
    else ("cuda:0" if torch.cuda.is_available() else "cpu")
)

MESH_PIPELINE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'models',
    device=DEVICE
)

PAINT_PIPELINE = Hunyuan3DPaintPipeline.from_pretrained(
    'models',
    subfolder='hunyuan3d-paint-v2-0-turbo',
)

print("[INFO] Models loaded successfully.")

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(force=True)
    concept = data.get("concept")
    method = data.get("method", "genai")

    if not concept:
        return jsonify({"error": "Missing 'concept' field"}), 400

    try:
        result_paths = generate_3d(
            concept=concept,
            method=method,
            output_dir="output_assets"
        )
        return jsonify({"status": "done", "results": result_paths}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    # Disable reloader to avoid duplicate processes
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
