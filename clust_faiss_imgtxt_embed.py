# generate_openclip_faiss_index_with_kmeans.py

import os
import torch
import faiss
import open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import numpy as np
import json

# ✅ 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ OpenCLIP 모델 로드
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# ✅ 경로 설정
image_root_dir = r"C:/크롤링 최종/all data"  # 🔥 이미지 폴더
embedding_save_path = r"C:/Users/user/Desktop/clust_faiss_imgtxt.pth"
faiss_index_save_path = r"C:/Users/user/Desktop/clust_faiss_imgtxt.index"
cluster_save_path = r"C:/Users/user/Desktop/clust_kmeans_clusters.json"

# ✅ 임베딩 생성
features = []
image_paths = []

image_files = [f for f in os.listdir(image_root_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

for filename in tqdm(image_files, desc="OpenCLIP 이미지+텍스트 임베딩 생성 중", ncols=80, dynamic_ncols=True):
    img_path = os.path.join(image_root_dir, filename)
    txt_path = os.path.splitext(img_path)[0] + ".txt"  # 같은 이름의 .txt 파일 찾기

    try:
        # ✅ 이미지 임베딩
        image = Image.open(img_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feat = model.encode_image(image_tensor)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        # ✅ 텍스트 임베딩 (없으면 0벡터 대체)
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            text_tokens = tokenizer([text]).to(device)
            with torch.no_grad():
                text_feat = model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        else:
            text_feat = torch.zeros((1, model.visual.output_dim)).to(device)

        # ✅ 이미지+텍스트 결합 (평균)
        combined_feat = (image_feat + text_feat) / 2
        features.append(combined_feat.squeeze(0).cpu())
        image_paths.append(img_path)

    except Exception as e:
        tqdm.write(f"❌ 실패: {img_path} ({str(e)})")
        continue

# ✅ 저장
if features:
    final_tensor = torch.stack(features)

    # 1️⃣ 임베딩과 이미지 경로 저장
    torch.save({"features": final_tensor, "paths": image_paths}, embedding_save_path)
    print(f"\n🎉 이미지+텍스트 임베딩 저장 완료 → {embedding_save_path}")

    # 2️⃣ FAISS 인덱스 생성 및 저장
    dim = final_tensor.shape[1]
    index = faiss.IndexFlatIP(dim)  # 내적 기반 (코사인 유사도)
    normalized_features = final_tensor / final_tensor.norm(dim=-1, keepdim=True)
    index.add(normalized_features.numpy())
    faiss.write_index(index, faiss_index_save_path)
    print(f"🎉 FAISS 인덱스 저장 완료 → {faiss_index_save_path}")

    # 3️⃣ KMeans 클러스터링
    n_clusters = 13  # 🔧 필요에 따라 조절 가능
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(final_tensor.numpy())

    # ✅ 클러스터별 결과 저장
    cluster_result = {}
    for label, path in zip(cluster_labels, image_paths):
        cluster_result.setdefault(int(label), []).append(path)

    with open(cluster_save_path, "w", encoding="utf-8") as f:
        json.dump(cluster_result, f, ensure_ascii=False, indent=2)
    print(f"📊 KMeans 클러스터링 결과 저장 완료 → {cluster_save_path}")
else:
    print("❌ 생성된 임베딩이 없습니다.")
