# preprocess_pdfs.py
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path
import pdfplumber
from typing import List, Dict, Optional

def norm(x: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (x or "")).strip()

def parse_credit_hours(text: str) -> Dict[str, Optional[float]]:
    if not text:
        return {"학점": None, "설계학점": None, "주당시간": None}
    
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    result = {"학점": None, "설계학점": None, "주당시간": None}
    
    if len(numbers) >= 1:
        result["학점"] = float(numbers[0])
    if len(numbers) >= 2:
        result["설계학점"] = float(numbers[1])
    if len(numbers) >= 3:
        result["주당시간"] = float(numbers[2])
    
    return result

def get_table_type(tbl: list) -> Optional[str]:
    if not tbl or not tbl[0]:
        return None
    
    head = " ".join(norm(c) for c in tbl[0]).lower()
    body = " ".join(norm(c) for r in tbl for c in r if c).lower()
    
    if "강좌명" in head:
        return "basic"
    if "교육목표" in head:
        return "goals"
    if "평가항목" in head:
        return "eval"
    if "주요교재" in head or "주요교재" in body:
        return "books&notes"
    if "주" in head:
        return "weekly"
    
    return None

def collect_tables(pdf_path: Path) -> Dict[str, list]:
    buckets = {"basic": [], "goals": [], "eval": [], "books&notes": [], "weekly": []}
    
    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
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
    
    fallback_settings = {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 5,
        "join_tolerance": 5,
    }
    
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables(table_settings) or []
            
            if not tables:
                tables = page.extract_tables(fallback_settings) or []
            
            for tbl in tables:
                if not tbl or not tbl[0]:
                    continue
                
                table_type = get_table_type(tbl)
                if table_type:
                    buckets[table_type].append(tbl)

    return buckets

def parse_basic(tables):
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
            if not item:
                continue
            
            ratio = norm(row[2]) if len(row) > 2 and row[2] else ""
            if ratio:
                try:
                    ratio_val = float(ratio)
                    d[item] = ratio_val
                except ValueError:
                    pass
    
    return d

def parse_notes(tables):
    res = {"주교재": [], "참고교재": [], "학습준비사항": None, "수강학생 유의사항": None}
    
    def split_and_clean_textbooks(text: str) -> list:
        if not text:
            return []
        
        patterns = [r'\*주교재/', r'\*부교재/', r'\*참고교재/', r'기타\([^)]+\)/']
        combined_pattern = '|'.join(patterns)
        matches = list(re.finditer(combined_pattern, text))
        
        if not matches:
            return [text.strip()] if text.strip() else []
        
        books = []
        current_pos = 0
        
        for i, match in enumerate(matches):
            if i > 0:
                book_text = text[current_pos:match.start()].strip()
                if book_text:
                    books.append(book_text)
            current_pos = match.end()
        
        last_book = text[current_pos:].strip()
        if last_book:
            books.append(last_book)
        
        return books
    
    for t in tables:
        for i, row in enumerate(t):
            if not row:
                continue
            
            cells = [norm(cell) if cell else "" for cell in row]
            
            if len(cells) < 1:
                continue
            
            col0 = cells[0] if len(cells) > 0 else ""
            col1 = cells[1] if len(cells) > 1 else ""
            col2 = cells[2] if len(cells) > 2 else ""
            
            if "주교재" in col1:
                res["주교재"] = split_and_clean_textbooks(col2)
                continue

            if "참고교재" in col1:
                res["참고교재"] = split_and_clean_textbooks(col2)
                continue
            
            if "학습준비사항" in col0:
                res["학습준비사항"] = col1
                continue
            
            if "수강학생 유의" in col0:
                res["수강학생 유의사항"] = col1
                continue
    
    if not res["주교재"]:
        res["주교재"] = None
    if not res["참고교재"]:
        res["참고교재"] = None
    
    return res

def parse_weekly(tables):
    weekly_dict = {}
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

            weekly_dict[f"{wn}주차"] = {
                "핵심어": keyword,
                "세부내용": desc,
                "교수방법": method
            }
    
    return weekly_dict

def process_single_pdf(pdf_path: Path) -> Dict:
    buckets = collect_tables(pdf_path)

    info = parse_basic(buckets["basic"])
    goals = parse_goals(buckets["goals"])
    eval_dict = parse_eval(buckets["eval"])
    notes = parse_notes(buckets["books&notes"])
    weekly_dict = parse_weekly(buckets["weekly"])

    abeek_info = {}
    if "교과영역" in info or "교과영역(*) (ABEEK Classification)" in info:
        abeek_info["교과영역(*) (ABEEK Classification)"] = info.get("교과영역") or info.get("교과영역(*) (ABEEK Classification)")
    if "인증구분" in info or "인증구분(*) (ABEEK Requirement)" in info:
        abeek_info["인증구분(*) (ABEEK Requirement)"] = info.get("인증구분") or info.get("인증구분(*) (ABEEK Requirement)")

    credit_hours = parse_credit_hours(info.get("학점/주당시간", ""))

    metadata = {
        "source_pdf": pdf_path.name,
        "강좌명": info.get("강좌명"),
        "담당교수": info.get("담당교수"),
        "년도": info.get("년도"),
        "학기": info.get("학기"),
        "분반": info.get("분반"),
        "수강대상학과": info.get("수강대상학과"),
        "학점": credit_hours["학점"],
        "설계학점": credit_hours["설계학점"],
        "주당시간": credit_hours["주당시간"],
        "성적스케일": info.get("성적스케일"),
        "교과목유형": info.get("교과목유형"),
        "연락처": info.get("연락처"),
        "강좌형식": info.get("강좌형식"),
        "공학인증 교과목 관련 항목": abeek_info if abeek_info else None,
        "필수 선수과목": info.get("필수 선수과목"),
        "권장 선수과목": info.get("권장 선수과목"),
        "교과목 개요": info.get("교과목 개요"),
        "교육목표": goals,
        "평가항목": eval_dict,
        "주요교재": notes.get("주교재"),
        "참고교재(대표)": notes.get("참고교재"),
        "학습준비사항": notes.get("학습준비사항"),
        "수강학생 유의 및 참고사항": notes.get("수강학생 유의사항"),
        "주차별 강의개요": weekly_dict
    }

    return metadata

def process_all_pdfs(
    input_dir: Path = Path("data/pdfs"),
    output_path: Path = Path("data/processed/course_chunks.json")
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    all_courses = []
    pdf_files = sorted(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"⚠ 경고: {input_dir}에 PDF 파일이 없습니다.")
        return
    
    chunk_id = 0
    for pdf_path in pdf_files:
        if "설계교육계획서" in pdf_path.name:
            print(f"⏭ Skipping: {pdf_path.name} (설계교육계획서)")
            continue
            
        print(f"▶ Processing: {pdf_path.name}")
        try:
            course_data = process_single_pdf(pdf_path)
            course_data["chunk_id"] = chunk_id
            chunk_id += 1
            all_courses.append(course_data)
            print(f"  ✅ 메타데이터 생성 완료 (chunk_id: {course_data['chunk_id']})")
        except Exception as e:
            print(f"  ⚠ {pdf_path.name} 처리 중 오류: {e}")

    if all_courses:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_courses, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 총 {len(all_courses)}개 강의 저장 → {output_path}")
    else:
        print("\n⚠ 처리된 데이터가 없습니다.")

if __name__ == "__main__":
    process_all_pdfs()