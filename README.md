# 🛒 대화형 상품 검색

멀티모달 검색과 대화형 질의응답을 결합한 상품 검색 시스템입니다.  
**OpenCLIP + FAISS + GPT** 조합을 활용해 사용자의 질의를 이해하고,  
이미지/텍스트 임베딩 기반으로 유사한 상품을 추천합니다.

---

## ✨ 주요 기능
- 🔍 **PlugIR 3단계 질의 처리**: 요약 → 질문 생성 → 질문 필터링
- 🖼 **멀티모달 검색**: 이미지 + 텍스트 임베딩 결합
- 💬 **대화형 질의응답**: Streamlit 기반 UI
- ⚡ **빠른 검색**: FAISS를 활용한 실시간 후보 검색
- 💾 **임베딩 캐시**: 임베딩 저장/로드로 속도 최적화

---

## 🛠 기술 스택
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![OpenCLIP](https://img.shields.io/badge/OpenCLIP-blue)
![FAISS](https://img.shields.io/badge/FAISS-black)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?logo=streamlit&logoColor=white)
![GPT](https://img.shields.io/badge/GPT-API-lightgrey)

---

## 📂 프로젝트 구조  
📦 PlugIR-inspired-CLIP-based-Interactive-Product-Search  
 ┣ 📂 data           # 예시 이미지/텍스트 데이터  
 ┣ 📂 src            # 소스 코드  
 ┣ 📂 assets         # 데모 이미지/GIF  
 ┣ 📜 requirements.txt  
 ┣ 📜 app.py         # Streamlit 메인 앱  
 ┗ 📜 README.md  


## ​ 프로젝트 아키텍처 & 데모 화면

### 시스템 아키텍처
![시스템 구조](System_Architecture_image.png)

### 데모 UI 예시
![데모 화면](demo_image.png)

## 📚 참고 자료
- [PlugIR 논문](https://arxiv.org/abs/2406.03411)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [FAISS](https://github.com/facebookresearch/faiss)

## 📬 연락처
- **개발자**: 태훈 (yth91111@naver.com)
- **GitHub**: [taehoon1223](https://github.com/taehoon1223)
