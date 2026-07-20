"""
SnapCoord - Webアプリ版
服の写真をアップロードすると、似ているコーデ写真を表示する
"""

import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer, util
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)

# アップロードされた写真の一時保存先
UPLOAD_FOLDER = "uploads"
DATASET_DIR = "dataset"
TOP_N = 5

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("モデルを読み込んでいます...")
model = SentenceTransformer("clip-ViT-B-32")


def get_color_histogram(image, resize_to=(100, 100)):
    img_small = image.resize(resize_to)
    hist = np.array(img_small.histogram(), dtype=np.float32)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def color_similarity(hist1, hist2):
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(hist1, hist2) / (norm1 * norm2))


def image_to_base64(image_path):
    """画像ファイルをbase64に変換してブラウザで表示できるようにする"""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((400, 400))
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def load_dataset():
    """datasetフォルダの画像を全部読み込む"""
    valid_ext = (".jpg", ".jpeg", ".png")
    paths = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.lower().endswith(valid_ext)
    ]
    images = [Image.open(p).convert("RGB") for p in paths]
    return paths, images


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    if "file" not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "ファイルが選択されていません"}), 400

    # アップロードされた写真を一時保存
    filename = secure_filename(file.filename)
    query_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(query_path)

    # datasetを読み込む
    paths, images = load_dataset()
    if not images:
        return jsonify({"error": "datasetフォルダに画像がありません"}), 400

    # 形の類似度(CLIP)
    dataset_embeddings = model.encode(
        images, convert_to_tensor=True, show_progress_bar=False
    )
    query_image = Image.open(query_path).convert("RGB")
    query_embedding = model.encode(query_image, convert_to_tensor=True)
    clip_scores = util.cos_sim(query_embedding, dataset_embeddings)[0].cpu().numpy()

    # 色の類似度
    dataset_hists = [get_color_histogram(img) for img in images]
    query_hist = get_color_histogram(query_image)
    color_scores = np.array([color_similarity(query_hist, h) for h in dataset_hists])

    # 組み合わせスコア
    combined = 0.6 * clip_scores + 0.4 * color_scores
    top_indices = np.argsort(combined)[::-1][: min(TOP_N, len(images))]

    # 結果を画像データとして返す
    results = []
    for idx in top_indices:
        results.append(
            {
                "image": image_to_base64(paths[idx]),
                "score": round(float(combined[idx]), 3),
                "filename": os.path.basename(paths[idx]),
            }
        )

    # アップロード写真もbase64で返す(プレビュー用)
    query_b64 = image_to_base64(query_path)

    return jsonify({"query": query_b64, "results": results})


if __name__ == "__main__":
    app.run(debug=True)
