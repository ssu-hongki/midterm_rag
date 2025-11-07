import pdfplumber
import json
import re
import os
import sys
from typing import List, Dict, Any

# --------------------------------------------------------------------------------
# A. 데이터 구조화 함수
# --------------------------------------------------------------------------------

def clean_text(text: Any) -> str:
    """텍스트 정리: None 처리, 줄바꿈, 다중 공백 제거"""
    if text is None:
        return ""
    text = str(text)
    # 줄바꿈 및 다중 공백 제거
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_overview_chunk(text_raw: str) -> (Dict[str, Any], str):
    """
    [수정됨] 페이지 1 텍스트에서 강의개요 정보를 추출하여 
    청크(Dict)와 과목명(str)을 반환합니다.
    """
    
    # 1. 키 정보 추출
    course_title = instructor = year = semester = course_no = ""
    prerequisites = course_description = main_textbook = ""
    assessments = {}
    
    # 년도 학기 과목코드 (예: 2025학년도 2학기 2150132404)
    match_ycs = re.search(r'(\d{4})학년도 (\d)학기 (\d+)', text_raw)
    if match_ycs:
        year, semester, course_no = match_ycs.groups()
        
    # 강좌명 담당교수 (예: 데이터베이스 정영희)
    match_title_inst = re.search(r'강좌명 담당교수\n(\S+)\s+(\S+)', text_raw)
    if match_title_inst:
        course_title, instructor = match_title_inst.groups()
    
    # (예외) 다른 양식 (예: 자연언어처리 김성신)
    if not course_title:
        match_title = re.search(r'강좌명\s+(\S+)\s+담당교수', text_raw)
        if match_title: course_title = match_title.group(1)
    if not instructor:
        match_inst = re.search(r'담당교수\s+(\S+)\s+', text_raw)
        if match_inst: instructor = match_inst.group(1)

    # 권장 선수과목
    match_pre = re.search(r'권장 선수과목\n(.+)', text_raw)
    if match_pre:
        prerequisites = match_pre.group(1).strip()
        
    # 교과목 개요 (Course Description)
    match_desc = re.search(r'교과목 개요\n(.+?)\n교육목표', text_raw, re.DOTALL)
    if match_desc:
        course_description = clean_text(match_desc.group(1).replace('(Course Description)', '').strip())

    # 주교재 (예: *주교재/데이터베이스 for Beginner/우재남/한빛아카데미)
    match_textbook = re.search(r'\*주교재/(.+)', text_raw)
    if match_textbook:
        main_textbook = clean_text(match_textbook.group(1).strip())
        
    # 평가 정보 추출 (예: 중간고사 35 100)
    match_assess = re.findall(r'(중간고사|기말고사|과제|출석)\s+(\d+)\s+100', text_raw)
    for item, ratio in match_assess:
        assessments[item] = ratio
        
    assessment_str = ", ".join([f"{k}: {v}%" for k, v in assessments.items()])

    # 2. 최종 청크 구성
    content = (
        f"본 강의는 **{course_title}** 과목으로 **{year}년 {semester}학기**에 개설됩니다. "
        f"담당 교수는 **{instructor}**입니다. 과목 코드는 {course_no}이며 3학점(3시간) 강좌입니다. "
        f"권장 선수과목은 **{prerequisites}**입니다. "
        f"주요 교재는 **{main_textbook}**입니다. "
        f"교과목 개요: {course_description} "
        f"성적 평가 비율은 {assessment_str}입니다."
    )
    
    overview_chunk = {
        "title": "강의개요",
        "content": content,
        "metadata": {"type": "overview", "course_title": course_title}
    }
    
    # [수정됨] 과목명을 반환하여 주차별 청크에 주입할 수 있도록 함
    return overview_chunk, course_title

def extract_weekly_syllabus_chunks(
    table_data: List[List[str]],
    course_title: str # [수정됨] 과목명을 받음
) -> List[Dict[str, Any]]:
    """테이블 데이터에서 주차별 강의개요 청크 생성"""
    weekly_chunks = []
    
    rows = table_data[1:]
    
    for row in rows:
        if len(row) < 5: continue 
        
        week = clean_text(row[0])
        keyword = clean_text(row[1])
        description = clean_text(row[2])
        texts = clean_text(row[4])
        
        week_match = re.match(r'(\d{1,2})', week) # 숫자로 시작하는지 확인
        if not week_match: continue # 주차 번호가 아닌 행 스킵
            
        week_num_str = week_match.group(1)
            
        # 1. [수정됨] 내용 구성 (과목명 문맥 주입)
        content = f"과목: {course_title}\n{week_num_str}주차는"
        if keyword:
            content += f" 키워드 **{keyword}**로, {description}에 대해 학습합니다."
        else:
            content += f" {description}에 대해 학습합니다."
            
        if texts:
            content += f" (교재범위: {texts})"

        # 2. 제목 구성
        title = f"주차별 강의개요: {week_num_str}주차"
        if "중간고사" in description:
            title += " (중간고사)"
        elif "기말고사" in description:
            title += " (기말고사)"

        # 3. [수정됨] 청크 추가 (week를 int로 저장)
        weekly_chunks.append({
            "title": title,
            "content": content,
            "metadata": {
                "type": "weekly", 
                "week": int(week_num_str), # [수정] "03" (str) -> 3 (int)
                "course_title": course_title
            }
        })
        
    return weekly_chunks

# --------------------------------------------------------------------------------
# B. 메인 파이프라인 함수
# --------------------------------------------------------------------------------

def process_single_syllabus(file_path: str) -> List[Dict[str, Any]] | None:
    """단일 PDF 파일을 처리하여 RAG 데이터 리스트를 반환"""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) < 2:
                print(f"  [경고] 파일 {file_path}: 페이지가 부족합니다. 스킵합니다.")
                return None
            
            page1_text_raw = pdf.pages[0].extract_text(x_tolerance=1)
            
            page2_table_data = pdf.pages[1].extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 3,
            })
            
            if not page1_text_raw or not page2_table_data:
                print(f"  [경고] 파일 {file_path}: 필요한 텍스트/테이블을 추출하지 못했습니다. 스킵합니다.")
                return None

            # 3. [수정됨] 데이터 구조화
            overview_chunk, course_title = extract_overview_chunk(page1_text_raw)
            
            if not course_title:
                 print(f"  [경고] {file_path}: '과목명'을 추출하지 못했습니다. (비어있는 JSON이 생성될 수 있음)")

            # [수정됨] 주차별 청크에 과목명 전달
            weekly_chunks = extract_weekly_syllabus_chunks(page2_table_data[0], course_title) 

            return [overview_chunk] + weekly_chunks

    except Exception as e:
        print(f"  [오류] 파일 {file_path} 처리 중 예외 발생: {e}")
        return None

def automate_syllabus_to_json(input_dir: str, output_file: str):
    """지정된 폴더의 모든 PDF 강의계획서를 JSON 파일로 변환하여 저장"""
    
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    all_rag_data = []
    
    if not os.path.exists(input_dir):
        print(f"❌ 오류: 입력 폴더 '{input_dir}'를 찾을 수 없습니다. 폴더를 생성하고 PDF 파일을 넣어주세요.")
        return

    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"⚠️ 경고: '{input_dir}' 폴더에 처리할 PDF 파일이 없습니다.")
        return

    print(f"--- 총 {len(pdf_files)}개의 PDF 파일 처리를 시작합니다 ---")
    
    for filename in pdf_files:
        file_path = os.path.join(input_dir, filename)
        print(f"➡️ 처리 중: {filename}")
        
        rag_data = process_single_syllabus(file_path)
        
        if rag_data:
            for chunk in rag_data:
                chunk['metadata']['source_file'] = filename
            all_rag_data.extend(rag_data)
            print(f"  [성공] {len(rag_data)}개의 청크 추가.")
        else:
            print(f"  [실패] {filename} 파일 처리 실패/스킵.")

    if all_rag_data:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_rag_data, f, ensure_ascii=False, indent=4)
            print(f"\n--- ✅ 모든 파일 처리 완료! ---")
            print(f"총 {len(all_rag_data)}개의 RAG 데이터 청크가 '{output_file}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n❌ JSON 파일 저장 중 오류 발생: {e}")
    else:
        print("\n❌ 처리된 데이터가 없어 JSON 파일이 생성되지 않았습니다.")

# --------------------------------------------------------------------------------
# C. 실행부
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. 입력 폴더와 출력 파일 설정
    # (주의!) RAG 프로젝트 폴더 구조에 맞게 경로를 'data/'로 지정합니다.
    INPUT_DIRECTORY = "data/pdfs" # PDF 파일들을 이 폴더 안에 넣으세요
    OUTPUT_FILE = "data/all_syllabi_rag_data.json" # (주의!) 이 파일이 RAG의 입력임

    # 2. 실행
    automate_syllabus_to_json(INPUT_DIRECTORY, OUTPUT_FILE)