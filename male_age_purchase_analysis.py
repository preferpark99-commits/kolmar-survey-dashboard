"""
남성 응답자 중 연령대별 구매 의향 분석
"""

import pandas as pd

# CSV 파일 읽기
df_raw = pd.read_csv('헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv', header=None)
df = df_raw[df_raw[0].str.contains('2026', na=False)].copy()
df.reset_index(drop=True, inplace=True)
df.columns = ['타임스탬프', '성별', '연령대', '머리감는시간', '두피고민', '샴푸선택이유', '샴푸아쉬운점', 'Q7', 'Q8', '기타1', '기타2']

# 남성 필터링
male_df = df[df['성별'] == '남성']
female_df = df[df['성별'] == '여성']

print('=' * 60)
print('📋 응답자 현황')
print('=' * 60)
print(f'전체 응답자: {len(df)}명')
print(f'  - 남성: {len(male_df)}명 ({len(male_df)/len(df)*100:.1f}%)')
print(f'  - 여성: {len(female_df)}명 ({len(female_df)/len(df)*100:.1f}%)')

print('\n' + '-' * 60)
print('📊 전체 응답자 연령대별 분포')
print('-' * 60)
age_counts = df['연령대'].value_counts().sort_index()
for age, count in age_counts.items():
    print(f'  {age}: {count}명 ({count/len(df)*100:.1f}%)')

print('\n' + '-' * 60)
print('📊 남성 응답자 연령대별 분포')
print('-' * 60)
male_age_counts = male_df['연령대'].value_counts().sort_index()
for age, count in male_age_counts.items():
    print(f'  {age}: {count}명 ({count/len(male_df)*100:.1f}%)')

print('\n' + '=' * 60)
print('💰 남성 응답자 중 연령대별 구매 의향 분석')
print('=' * 60)

# 20대 남성
male_20s = male_df[male_df['연령대'] == '20대'] # type: ignore
male_20s_yes = male_20s[male_20s['Q8'] == '있다']
male_20s_pct = len(male_20s_yes) / len(male_20s) * 100 if len(male_20s) > 0 else 0

print(f'\n📊 20대 남성:')
print(f'   전체: {len(male_20s)}명')
print(f'   구매 의향 있다: {len(male_20s_yes)}명')
print(f'   비율: {male_20s_pct:.1f}%')

# 30대 남성
male_30s = male_df[male_df['연령대'] == '30대'] # type: ignore
male_30s_yes = male_30s[male_30s['Q8'] == '있다']
male_30s_pct = len(male_30s_yes) / len(male_30s) * 100 if len(male_30s) > 0 else 0

print(f'\n📊 30대 남성:')
print(f'   전체: {len(male_30s)}명')
print(f'   구매 의향 있다: {len(male_30s_yes)}명')
print(f'   비율: {male_30s_pct:.1f}%')

# 20대 + 30대 합계
male_20_30s = male_df[male_df['연령대'].isin(['20대', '30대'])]
male_20_30s_yes = male_20_30s[male_20_30s['Q8'] == '있다']
male_20_30s_pct = len(male_20_30s_yes) / len(male_20_30s) * 100 if len(male_20_30s) > 0 else 0

print(f'\n📊 20~30대 남성 (합계):')
print(f'   전체: {len(male_20_30s)}명')
print(f'   구매 의향 있다: {len(male_20_30s_yes)}명')
print(f'   비율: {male_20_30s_pct:.1f}%')

print('\n' + '=' * 50)
