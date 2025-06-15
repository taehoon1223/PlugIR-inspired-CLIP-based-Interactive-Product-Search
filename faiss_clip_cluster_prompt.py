# streamlit run "C:/Users/user/Desktop/faiss_clip_cluster_prompt.py"


import os
import torch
import faiss
import streamlit as st
import open_clip
from openai import OpenAI
import time
import json

# ✅ 환경 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
client = OpenAI(api_key="")  # 🔥 API 키 입력 필요

# ✅ OpenCLIP 모델 로드
model, preprocess, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# ✅ 데이터 로드
embedding_data = torch.load(r"C:/Users/user/Desktop/embedding/clust_faiss_img.pth", map_location=device)
#embedding_data = torch.load(r"C:/Users/user/Desktop/embedding/clust_faiss_imgtxt.pth", map_location=device)
all_features = embedding_data['features'].to(device)
image_paths = embedding_data['paths']
faiss_index = faiss.read_index(r"C:/Users/user/Desktop/embedding/clust_faiss_img.index")
#faiss_index = faiss.read_index(r"C:/Users/user/Desktop/embedding/clust_faiss_imgtxt.index")

# ✅ 클러스터링 결과 로드 (JSON 기반)
with open(r"C:/Users/user/Desktop/embedding/clust_faiss_img.json", "r", encoding="utf-8") as f:
#with open(r"C:/Users/user/Desktop/embedding/clust_faiss_imgtxt.json", "r", encoding="utf-8") as f:
    cluster_data = json.load(f)
path_to_cluster = {p: int(cid) for cid, paths in cluster_data.items() for p in paths}

# ✅ 프롬프트 함수들
def plugir_summarize(context_text):
    prompt = f"""
    You are a shopping assistant. Summarize the user's intent for image retrieval.
    Dialogue history:
    {context_text.strip()}
    Respond with only one sentence summary.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def plugir_generate_question(summary, cluster_description):
    prompt = f"""
    You are an assistant helping refine product search.
    Summarized Query: {summary}
    Cluster Description: {cluster_description}
    Ask one specific follow-up question to narrow down the search within this cluster.
    Respond only with the question.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

def plugir_filter_question(summary, question, context_text):
    prompt = f"""
    Summarized Query: {summary}
    Candidate Question: {question}
    Already Known: {context_text.strip()}
    You can't ask questions that are common to previous ones.
    Decide if the question is Useful or Unnecessary.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful shopping assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ✅ 필터링 실패 시 재시도 함수
def generate_valid_question(summary, cluster_description, context_text, max_tries=3):
    for _ in range(max_tries):
        question = plugir_generate_question(summary, cluster_description)
        result = plugir_filter_question(summary, question, context_text)
        if result.lower().startswith("useful"):
            return question
    return None

# ✅ 텍스트 임베딩 함수
def get_openclip_text_embedding(text):
    templates = [
        f"a photo of a {text} (color emphasized)",
        f"a clear image showing {text} color",
        f"a close-up photo of a {text}",
        f"{text} on a plain background",
        f"an object described as {text}"
    ]
    with torch.no_grad():
        tokens = tokenizer(templates).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.mean(dim=0, keepdim=True)

# ✅ 검색 함수
def search_faiss(context_embedding, top_k=10):
    normalized_query = context_embedding / context_embedding.norm(dim=-1, keepdim=True)
    D, I = faiss_index.search(normalized_query.cpu().numpy(), top_k)
    return I[0], D[0]

# ✅ 세션 초기화
if "context_text" not in st.session_state:
    st.session_state.context_text = ""
if "image_history" not in st.session_state:
    st.session_state.image_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_timing" not in st.session_state:
    st.session_state.last_timing = None

# ✅ 페이지 UI 설정
st.set_page_config(page_title="🛍️ PlugIR+FAISS 쇼핑 도우미", layout="wide")
st.markdown("<h1 style='text-align: center;'>🛍️ Chatbot과 CLIP을 이용한 온라인 쇼핑몰 상품 검색</h1>", unsafe_allow_html=True)
st.markdown("---")
left_col, right_col = st.columns([1, 2])

# ✅ 왼쪽 입력 UI
with left_col:
    st.markdown("### 💬 대화")
    with st.form("chat_form", clear_on_submit=True):
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div style='text-align: right; background-color: #e0f7fa; padding: 6px; border-radius: 6px; margin: 4px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; background-color: #f1f1f1; padding: 6px; border-radius: 6px; margin: 4px 0;'>{msg['content']}</div>", unsafe_allow_html=True)
        user_input = st.text_input("메시지를 입력하세요", label_visibility="collapsed")
        submitted = st.form_submit_button("전송")

    if submitted and user_input.strip():
        st.session_state.messages = st.session_state.messages[-4:]
        st.session_state.context_text += " " + user_input.strip()
        st.session_state.messages.append({"role": "user", "content": user_input})

        total_start = time.perf_counter()

        summarize_start = time.perf_counter()
        summary_text = plugir_summarize(st.session_state.context_text)
        summarize_end = time.perf_counter()

        embedding_start = time.perf_counter()
        context_embedding = get_openclip_text_embedding(summary_text)
        embedding_end = time.perf_counter()

        search_start = time.perf_counter()
        indices, scores = search_faiss(context_embedding, top_k=10)
        search_end = time.perf_counter()

        retrieved_paths = [image_paths[i] for i in indices]
        retrieved_clusters = [path_to_cluster.get(p, -1) for p in retrieved_paths]
        major_cluster = max(set(retrieved_clusters), key=retrieved_clusters.count)
        cluster_description = f"Cluster {major_cluster}"

        question_start = time.perf_counter()
        final_question = generate_valid_question(summary_text, cluster_description, st.session_state.context_text)
        question_end = time.perf_counter()

        images = [(image_paths[i], scores[idx]) for idx, i in enumerate(indices)]
        st.session_state.image_history.append(images)

        st.session_state.last_timing = {
            "summarize_time": summarize_end - summarize_start,
            "embedding_time": embedding_end - embedding_start,
            "search_time": search_end - search_start,
            "question_time": question_end - question_start,
            "total_time": time.perf_counter() - total_start
        }

        st.session_state.messages.append({"role": "assistant", "content": f"Summary: {summary_text}"})
        if final_question:
            st.session_state.messages.append({"role": "assistant", "content": final_question})

        st.rerun()

# ✅ 오른쪽 이미지 결과 출력
with right_col:
    st.markdown("### 🔍 추천 이미지 Top 10")
    if st.session_state.image_history:
        imgs = st.session_state.image_history[-1]
        cols = st.columns(5)
        for idx, (img_path, score) in enumerate(imgs):
            with cols[idx % 5]:
                img_name = os.path.basename(img_path)
                st.image(img_path, use_container_width=True)
                st.markdown(f"<div style='text-align: center; font-size: 14px;'>📄 {img_name}<br>🔍 유사도: {score:.2f}</div>", unsafe_allow_html=True)

    if st.session_state.last_timing:
        t = st.session_state.last_timing
        st.success(f"""
            ⏱️ **처리 시간 요약 (최근 기준):**
            - 요약: {t['summarize_time']:.2f}초
            - 임베딩: {t['embedding_time']:.2f}초
            - 검색: {t['search_time']:.2f}초
            - 질문 생성: {t['question_time']:.2f}초
            - 전체 시간: {t['total_time']:.2f}초
        """)

