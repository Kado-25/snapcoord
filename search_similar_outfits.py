"""
似ているコーデ写真を検索するスクリプト(形 + 色 を組み合わせて判定)

仕組み:
1. dataset フォルダの中にある参考コーデ写真を全部「特徴ベクトル(形)」と「色ヒストグラム(色)」に変換する
2. query.png(自分の服の写真)も同じように変換する
3. 「形の近さ」と「色の近さ」を組み合わせたスコアで、似ているコーデを上位5枚表示する

事前準備:
    pip install sentence-transformers pillow torch

フォルダ構成:
    fashion_search/
        search_similar_outfits.py   <- このファイル
        dataset/                    <- 参考コーデ写真をここに入れる(jpg, png)
            outfit1.jpg
            outfit2.jpg
            ...
        query.png                   <- 自分が着たい服の写真
"""

import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

DATASET_DIR = "dataset"
QUERY_IMAGE = "query.png"
TOP_N = 5  # 上位何枚を表示するか

# 「形の近さ」と「色の近さ」をどれくらい重視するか(合計が1になるように調整する)
CLIP_WEIGHT = 0.4   # 形・カテゴリの近さの重み
COLOR_WEIGHT = 0.6  # 色の近さの重み


def load_images_from_folder(folder_path):
    """フォルダ内の画像ファイルを全部読み込む"""
    valid_ext = (".jpg", ".jpeg", ".png")
    paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(valid_ext)
    ]
    images = [Image.open(p).convert("RGB") for p in paths]
    return paths, images


def get_color_histogram(image, resize_to=(100, 100)):
    """
    画像の「色の特徴」を数値化する。
    画像を縮小してRGBごとのヒストグラム(256段階 x 3色 = 768個の数値)を作り、
    画像サイズに依存しないよう合計が1になるように正規化する。
    """
    img_small = image.resize(resize_to)
    hist = np.array(img_small.histogram(), dtype=np.float32)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def color_similarity(hist1, hist2):
    """2つの色ヒストグラムのコサイン類似度を計算する(1に近いほど色が似ている)"""
    norm1 = np.linalg.norm(hist1)
    norm2 = np.linalg.norm(hist2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(hist1, hist2) / (norm1 * norm2))


def main():
    print("モデルを読み込んでいます(初回は少し時間がかかります)...")
    # CLIPという画像とテキストの両方を理解できるAIモデルを使う(形・カテゴリの判定が得意)
    model = SentenceTransformer("clip-ViT-B-32")

    if not os.path.isdir(DATASET_DIR):
        print(f"エラー: '{DATASET_DIR}' フォルダが見つかりません。参考コーデ写真を入れてください。")
        return

    paths, images = load_images_from_folder(DATASET_DIR)
    if not images:
        print(f"エラー: '{DATASET_DIR}' フォルダに画像がありません。")
        return

    if not os.path.isfile(QUERY_IMAGE):
        print(f"エラー: '{QUERY_IMAGE}' が見つかりません。検索したい服の写真を置いてください。")
        return

    print(f"{len(images)}枚の参考コーデ写真を解析しています(形)...")
    dataset_embeddings = model.encode(images, convert_to_tensor=True, show_progress_bar=False)

    print(f"{len(images)}枚の参考コーデ写真を解析しています(色)...")
    dataset_histograms = [get_color_histogram(img) for img in images]

    print("自分の服の写真を解析しています...")
    query_image = Image.open(QUERY_IMAGE).convert("RGB")
    query_embedding = model.encode(query_image, convert_to_tensor=True)
    query_histogram = get_color_histogram(query_image)

    # 形の近さ(CLIP)を計算する
    clip_similarities = util.cos_sim(query_embedding, dataset_embeddings)[0].cpu().numpy()

    # 色の近さを1枚ずつ計算する
    color_similarities = np.array(
        [color_similarity(query_histogram, h) for h in dataset_histograms]
    )

    # 形と色を重み付きで組み合わせた最終スコアを作る
    combined_scores = CLIP_WEIGHT * clip_similarities + COLOR_WEIGHT * color_similarities

    # スコアが高い順に並べて上位N枚を取り出す
    top_n = min(TOP_N, len(images))
    top_indices = np.argsort(combined_scores)[::-1][:top_n]

    print(f"\n--- 似ているコーデ Top {TOP_N} (形{int(CLIP_WEIGHT*100)}% + 色{int(COLOR_WEIGHT*100)}%) ---")
    for rank, idx in enumerate(top_indices, start=1):
        print(
            f"{rank}. {paths[idx]}  "
            f"(総合: {combined_scores[idx]:.3f}  "
            f"形: {clip_similarities[idx]:.3f}  "
            f"色: {color_similarities[idx]:.3f})"
        )


if __name__ == "__main__":
    main()
