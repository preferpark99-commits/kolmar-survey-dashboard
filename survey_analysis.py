import pandas as pd

# CSV 파일 읽기 (헤더 없이 전체 읽은 후 실제 데이터만 필터링)
df_raw = pd.read_csv('헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv', header=None)

# 실제 응답 데이터만 필터링 (타임스탬프에 '2026'이 포함된 행)
df = df_raw[df_raw[0].str.contains('2026', na=False)].copy()
df.reset_index(drop=True, inplace=True)

# 컬럼명 직접 지정 (원본 헤더 기반) - 11개 컬럼
df.columns = ['타임스탬프', '성별', '연령대', '머리감는시간', '두피고민', '샴푸선택이유', '샴푸아쉬운점', 'Q7', 'Q8', '기타1', '기타2']

# 컬럼명 확인을 위해 출력
print("컬럼명:", df.columns.tolist())
print()

# 남성이면서 Q8에 "있다"로 응답한 사람 필터링
male_with_purchase_intent = df[(df['성별'] == '남성') & (df['Q8'] == '있다')]

# 결과 출력
total_count = len(df)
male_count = len(df[df['성별'] == '남성'])
male_purchase_intent_count = len(male_with_purchase_intent)

print(f"전체 응답자 수: {total_count}명")
print(f"남성 응답자 수: {male_count}명 ({male_count/total_count*100:.1f}%)")
print(f"남성 중 구매 의향 '있다' 응답자 수: {male_purchase_intent_count}명 ({male_purchase_intent_count/male_count*100:.1f}%)")
