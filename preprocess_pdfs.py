# preprocess_pdfs.py
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path
import pdfplumber
import tempfile
import os

try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("⚠ 경고: gdown이 설치되지 않았습니다. 구글 드라이브 다운로드 기능을 사용하려면 'pip install gdown'을 실행하세요.")

# -------------------------------
# 공용 함수
# -------------------------------
def norm(x):
    return re.sub(r"\s+", " ", (x or "")).strip()

# -------------------------------
# 0) 표 유형 자동 판별
# -------------------------------
def is_weekly(tbl):
    head = " ".join(norm(c) for c in tbl[0])
    keys = ["주", "week", "핵심어", "keyword", "세부내용", "description"]
    return sum(k in head.lower() for k in keys) >= 2

def is_eval(tbl):
    head = " ".join(norm(c) for c in tbl[0]).lower()
    return ("평가항목" in head) and ("반영비율" in head)

def is_goals(tbl):
    head = " ".join(norm(c) for c in tbl[0]).lower()
    return ("교육목표" in head)

def is_texts(tbl):
    body = " ".join(norm(c) for r in tbl for c in r if c).lower()
    return any(k in body for k in ["주교재", "참고교재", "학습준비사항", "수강학생 유의"])

def is_basic(tbl):
    body = " ".join(norm(c) for r in tbl for c in r if c)
    keys = ["강좌명", "담당교수", "년도", "학기", "과목코드", "수강대상학과", "학점/주당시간", "이수구분"]
    return sum(k in body for k in keys) >= 3

# -------------------------------
# 1) 모든 페이지에서 표 수집 + 분류
# -------------------------------
def collect_tables(pdf_path: Path):
    buckets = {"basic": [], "goals": [], "eval": [], "texts": [], "weekly": []}
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "explicit_vertical_lines": [],
        "explicit_horizontal_lines": [],
        "snap_tolerance": 3, 
        "join_tolerance": 3, 
        "edge_min_length": 3, 
        "min_words_vertical": 3, 
        "min_words_horizontal": 1,
        "intersection_tolerance": 3,
        "text_tolerance": 3, 
        "text_x_tolerance": 3,
        "text_y_tolerance": 3,
    }
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables(table_settings) or []

            if not tables:
                fallback_settings = {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 5,
                    "join_tolerance": 5,
                }
                tables = page.extract_tables(fallback_settings) or []
            for tbl in tables:
                if not tbl or not tbl[0]:
                    continue
                if is_weekly(tbl):
                    buckets["weekly"].append(tbl)
                elif is_eval(tbl):
                    buckets["eval"].append(tbl)
                elif is_goals(tbl):
                    buckets["goals"].append(tbl)
                elif is_texts(tbl):
                    buckets["texts"].append(tbl)
                elif is_basic(tbl):
                    buckets["basic"].append(tbl)

    return buckets

# -------------------------------
# 2) 표 종류별 파서
# -------------------------------
def parse_basic_info(tables):
    info = {}
    for t in tables:
        for row in t:
            cells = [norm(c) if c else "" for c in row]
            for i in range(0, len(cells), 2):
                if i + 1 < len(cells):
                    key = re.sub(r"\(.*?\)", "", cells[i]).strip()
                    val = cells[i + 1]
                    if key and val:
                        info[key] = val
    return info

def parse_goals(tables):
    goals = []
    for t in tables:
        for row in t[1:]:
            if row and row[0]:
                goals.append(norm(row[0]))
    return goals

def parse_eval(tables):
    d = {}
    for t in tables:
        for row in t[1:]:
            if not row or len(row) < 3:
                continue
            item = norm(row[0]) if row[0] else ""
            max_s = norm(row[1]) if len(row) > 1 and row[1] else ""
            ratio = norm(row[2]) if len(row) > 2 and row[2] else ""
            
            max_score_val = None
            ratio_val = None
            
            if max_s:
                try:
                    max_score_val = float(max_s)
                except Exception:
                    pass
            
            if ratio:
                try:
                    ratio_val = float(ratio) / 100.0
                except Exception:
                    pass
            
            if item:
                d[item] = {"max_score": max_score_val, "ratio": ratio_val}
    return d

def parse_texts(tables):
    res = {"주교재": None, "참고교재": None, "학습준비사항": None, "수강학생 유의사항": None}
    for t in tables:
        for row in t:
            if not row or len(row) < 1:
                continue
            k = norm(row[0]) if row[0] else ""
            k = k.replace("\n", "")
            v = norm(row[1]) if len(row) > 1 and row[1] else ""

            if "주교재" in k:
                res["주교재"] = (res["주교재"] + "\n" if res["주교재"] else "") + v
            elif "참고교재" in k:
                res["참고교재"] = (res["참고교재"] + "\n" if res["참고교재"] else "") + v
            elif "학습준비사항" in k:
                res["학습준비사항"] = v
            elif "수강학생 유의" in k:
                res["수강학생 유의사항"] = v
    return res

def parse_weekly(tables):
    sents = []
    for t in tables:
        for row in t[1:]:
            if not row or len(row) < 2:
                continue
            
            week = norm(row[0]) if row[0] else ""
            wn = re.sub(r"[^0-9]", "", week)
            if not wn:
                continue

            keyword = norm(row[1]) if len(row) > 1 and row[1] else ""
            desc = norm(row[2]) if len(row) > 2 and row[2] else ""
            method = norm(row[3]) if len(row) > 3 and row[3] else ""
            texts = norm(row[4]) if len(row) > 4 and row[4] else ""

            if not keyword:
                continue
            
            sent = f"{wn}주차 강의 주제는 {keyword}입니다."
            if desc:
                sent += f" 주요 학습 내용은 {desc}입니다."
            if texts:
                sent += f" 교재범위는 {texts}입니다."
            if method:
                sent += f" 수업은 {method} 방식으로 진행됩니다."
            sents.append(sent)
    return sents

# -------------------------------
# 3) 청킹 (5문장 단위)
# -------------------------------
def chunk_by_sentences(sentences, chunk_size=5):
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunks.append(sentences[i:i + chunk_size])
    return chunks

# -------------------------------
# 4) 단일 PDF → 청크 리스트
# -------------------------------
def process_single_pdf(pdf_path: Path):
    buckets = collect_tables(pdf_path)

    info = parse_basic_info(buckets["basic"])
    goals = parse_goals(buckets["goals"])
    eval_dict = parse_eval(buckets["eval"])
    texts = parse_texts(buckets["texts"])
    weekly = parse_weekly(buckets["weekly"])

    sentences = []

    if "강좌명" in info:
        s = f"이 강의는 '{info['강좌명']}' 과목입니다."
        if "담당교수" in info:
            s += f" 담당 교수는 {info['담당교수']}입니다."
        sentences.append(s)

    if "년도" in info and "학기" in info:
        sentences.append(f"개설 학기는 {info['년도']} {info['학기']}입니다.")

    if "수강대상학과" in info:
        sentences.append(f"수강 대상은 {info['수강대상학과']}입니다.")

    if "교과목 개요" in info:
        sentences.append(f"교과목 개요: {info['교과목 개요']}")

    for g in goals:
        sentences.append(f"교육목표: {g}")

    sentences.extend(weekly)

    parts = [f"{k} {int(v['ratio']*100)}%" for k, v in eval_dict.items() if v["ratio"]]
    if parts:
        sentences.append("이 강의의 성적 평가는 " + ", ".join(parts) + "로 반영됩니다.")

    if texts["주교재"]:
        sentences.append(f"주교재는 {texts['주교재']}입니다.")
    if texts["참고교재"]:
        sentences.append(f"참고교재로는 {texts['참고교재']} 등이 사용됩니다.")
    if texts["학습준비사항"]:
        sentences.append(f"학습 준비 사항: {texts['학습준비사항']}.")
    if texts["수강학생 유의사항"]:
        sentences.append(f"수강 시 유의할 점: {texts['수강학생 유의사항']}.")

    chunks = chunk_by_sentences(sentences, chunk_size=5)

    # chunk 단위 데이터 구성
    chunk_payload = [
        {
            "chunk_id": i + 1,
            "source_pdf": pdf_path.name,
            "metadata": info,
            "text": "\n".join(chunk)
        }
        for i, chunk in enumerate(chunks)
    ]

    return chunk_payload

# -------------------------------
# 5) 구글 드라이브에서 PDF 다운로드
# -------------------------------
def is_google_drive_folder(url: str) -> bool:
    """URL이 구글 드라이브 폴더 링크인지 확인합니다."""
    return "/drive/folders/" in url.lower() or "/folders/" in url.lower()

def is_google_drive_url(url: str) -> bool:
    """URL이 구글 드라이브 링크인지 확인합니다."""
    return "drive.google.com" in url.lower()

def download_from_google_drive(google_drive_url: str, output_dir: Path = Path("data/pdfs")) -> list[Path]:
    """
    구글 드라이브 링크에서 PDF 파일 또는 폴더를 다운로드합니다.
    
    Args:
        google_drive_url: 구글 드라이브 공유 링크 
            - 파일: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
            - 폴더: https://drive.google.com/drive/folders/FOLDER_ID?usp=drive_link
        output_dir: 다운로드할 디렉토리
    
    Returns:
        다운로드된 파일들의 Path 객체 리스트
    """
    if not GDOWN_AVAILABLE:
        raise ImportError("gdown이 설치되지 않았습니다. 'pip install gdown'을 실행하세요.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []
    
    try:
        if is_google_drive_folder(google_drive_url):
            # 폴더 다운로드
            print(f"📁 구글 드라이브 폴더에서 다운로드 중...")
            # gdown의 download_folder는 폴더 ID를 사용
            # URL에서 폴더 ID 추출
            folder_id = None
            if "/folders/" in google_drive_url:
                parts = google_drive_url.split("/folders/")
                if len(parts) > 1:
                    folder_id = parts[1].split("?")[0].split("&")[0]
            
            if not folder_id:
                raise ValueError("폴더 ID를 추출할 수 없습니다. 올바른 구글 드라이브 폴더 링크를 제공하세요.")
            
            # 폴더 다운로드 (PDF 파일만)
            folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
            gdown.download_folder(folder_url, output=str(output_dir), quiet=False, use_cookies=False)
            
            # 다운로드된 PDF 파일 찾기
            for pdf_path in output_dir.glob("*.pdf"):
                if pdf_path not in downloaded_files:
                    downloaded_files.append(pdf_path)
                    print(f"  ✅ 다운로드 완료: {pdf_path.name}")
        else:
            # 단일 파일 다운로드
            print(f"📄 구글 드라이브 파일에서 다운로드 중...")
            # 임시 파일로 다운로드
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                temp_path = tmp_file.name
            
            try:
                gdown.download(google_drive_url, temp_path, fuzzy=True, quiet=False)
                
                # 파일명 추출
                filename = f"gdrive_{hash(google_drive_url) % 10000}.pdf"
                output_path = output_dir / filename
                
                # 임시 파일을 최종 위치로 이동
                os.rename(temp_path, str(output_path))
                downloaded_files.append(output_path)
                print(f"  ✅ 다운로드 완료: {output_path.name}")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        
        return downloaded_files
    except Exception as e:
        raise Exception(f"구글 드라이브 다운로드 실패: {e}")

# -------------------------------
# 6) 여러 PDF 한 번에 처리 (로컬 + 구글 드라이브 지원)
# -------------------------------
def process_all_pdfs(
    input_dir: Path = Path("data/pdfs"),
    output_path: Path = Path("data/processed/course_chunks.json"),
    google_drive_urls: list = None
):
    """
    로컬 폴더와 구글 드라이브에서 PDF를 처리합니다.
    
    Args:
        input_dir: 로컬 PDF 폴더 경로
        output_path: 출력 JSON 파일 경로
        google_drive_urls: 구글 드라이브 링크 리스트 (선택사항)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    
    # 1. 로컬 폴더의 PDF 처리
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        print(f"▶ Processing: {pdf_path.name}")
        try:
            chunks = process_single_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠ {pdf_path.name} 처리 중 오류: {e}")
    
    # 2. 구글 드라이브 링크 처리
    if google_drive_urls:
        if not GDOWN_AVAILABLE:
            print("⚠ 경고: gdown이 설치되지 않아 구글 드라이브 링크를 처리할 수 없습니다.")
        else:
            for url in google_drive_urls:
                if not is_google_drive_url(url):
                    print(f"⚠ 경고: 유효하지 않은 구글 드라이브 링크입니다: {url}")
                    continue
                
                try:
                    print(f"\n▶ 구글 드라이브에서 다운로드 중: {url}")
                    downloaded_paths = download_from_google_drive(url, input_dir)
                    
                    # 다운로드된 각 PDF 파일 처리
                    for downloaded_path in downloaded_paths:
                        try:
                            print(f"  ▶ Processing: {downloaded_path.name}")
                            chunks = process_single_pdf(downloaded_path)
                            all_chunks.extend(chunks)
                        except Exception as e:
                            print(f"    ⚠ {downloaded_path.name} 처리 중 오류: {e}")
                except Exception as e:
                    print(f"  ⚠ 구글 드라이브 링크 처리 중 오류 ({url}): {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {len(all_chunks)}개 청크 저장 → {output_path}")

if __name__ == "__main__":
    # 구글 드라이브 폴더 링크 사용
    google_drive_urls = [
        "https://drive.google.com/drive/folders/1fsqg3UR9RfNYJQQttrJRNpmT0gNXWyki?usp=drive_link"
    ]
    process_all_pdfs(google_drive_urls=google_drive_urls)
    
    # 또는 로컬 폴더만 처리하려면 위의 google_drive_urls 부분을 주석 처리하고 아래 주석을 해제하세요
    # process_all_pdfs()