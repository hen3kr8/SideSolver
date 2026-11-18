import base64
import logging
import os
import tempfile

import cv2
import puzzle_extractor
from flask import Flask, jsonify, render_template, request

MODEL_PATH = "models/finalized_model.keras"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    uploaded_file = request.files.get("puzzle_image")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify(error="No image was uploaded."), 400

    if not _allowed_file(uploaded_file.filename):
        return jsonify(error="Please upload a PNG or JPEG image."), 400

    suffix = os.path.splitext(uploaded_file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        solved_image = puzzle_extractor.solve_puzzle_from_image(tmp_path, MODEL_PATH)
    except Exception:
        logging.exception("failed to solve puzzle for uploaded image")
        return jsonify(error="Something went wrong while solving this puzzle."), 500
    finally:
        os.remove(tmp_path)

    if solved_image is None:
        return (
            jsonify(
                error=(
                    "Couldn't solve this puzzle. Make sure the full grid is "
                    "visible, well-lit, and roughly square-on, then try again."
                )
            ),
            422,
        )

    ok, encoded = cv2.imencode(".png", solved_image)
    if not ok:
        return jsonify(error="Failed to encode the solved image."), 500

    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return jsonify(image=f"data:image/png;base64,{image_b64}")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    app.run(debug=True, host="127.0.0.1", port=5000)
