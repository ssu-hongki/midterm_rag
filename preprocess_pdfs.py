# preprocess_pdfs.py
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path
import pdfplumber

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

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables() or []
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
            cells = [norm(c) for c in row if c]
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
            item = norm(row[0])
            max_s = norm(row[1])
            ratio = norm(row[2])
            try:
                max_s = float(max_s)
            except Exception:
                max_s = None
            try:
                ratio = float(ratio) / 100.0
            except Exception:
                ratio = None
            if item:
                d[item] = {"max_score": max_s, "ratio": ratio}
    return d

def parse_texts(tables):
    res = {"주교재": None, "참고교재": None, "학습준비사항": None, "수강학생 유의사항": None}
    for t in tables:
        for row in t:
            k = norm((row[0] or "")).replace("\n", "")
            v = norm(row[1]) if len(row) > 1 else ""
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
            week = norm(row[0])
            wn = re.sub(r"[^0-9]", "", week)
            if not wn:
                continue

            keyword = norm(row[1])
            desc = norm(row[2]) if len(row) > 2 else ""
            method = norm(row[3]) if len(row) > 3 else ""
            texts = norm(row[4]) if len(row) > 4 else ""

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
# 5) 여러 PDF 한 번에 처리
# -------------------------------
def process_all_pdfs(
    input_dir: Path = Path("data/pdfs"),
    output_path: Path = Path("data/processed/course_chunks.json")
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        print(f"▶ Processing: {pdf_path.name}")
        try:
            chunks = process_single_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ⚠ {pdf_path.name} 처리 중 오류: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {len(all_chunks)}개 청크 저장 → {output_path}")

if __name__ == "__main__":
    process_all_pdfs()