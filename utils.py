import re
from typing import List

# [JSON 파이프라인으로 변경]
# - 엑셀/CSV 스캔을 위한 COLUMN_DEFINITIONS와 헬퍼 함수들 (find_columns_by_aliases 등) 삭제.
# - chunking.py가 삭제되었으므로 WEEK_SPLIT_REGEX 삭제.

# --------------------------------------------------------------------------------
# 1. rag_chain.py가 "사용자 질문"을 파싱할 때 사용하는 정규식
# --------------------------------------------------------------------------------

# "3주", "Week 3", "제 5 주" 등을 인식 (숫자 그룹 1개)
# (data_loader.py가 아닌 rag_chain.py의 filtered_search에서 사용됨)
WEEK_NUM_REGEX = re.compile(
    r"(?:제\s*)?(\d{1,2})\s*(?:주|주차|week|차시)\b", 
    re.IGNORECASE
)

# --------------------------------------------------------------------------------
# 2. rag_chain.py가 "사용자 질문"을 확장할 때 사용하는 키워드
# --------------------------------------------------------------------------------

# "교재"라고 물었을 때 검색을 보강하기 위한 동의어
TEXTBOOK_KEYWORDS = [
    "교재", "주교재", "textbook", "required texts", "texts",
    "참고", "참고문헌", "참고자료", "references", "recommended reading"
]