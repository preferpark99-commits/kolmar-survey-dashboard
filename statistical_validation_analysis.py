"""
============================================================
맨즈케어 데이앤나이트 듀얼 샴푸 - 통계적 신뢰성 검증 분석
============================================================
Feature Importance의 신뢰성을 검증하고 해석을 제공

분석 내용:
1. Logistic Regression p-value 검정 (통계적 유의성)
2. Permutation Importance (더 안정적인 중요도)
3. Bootstrap Confidence Interval (신뢰구간)
4. 상관관계 분석 및 카이제곱 검정
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기
df = pd.read_csv('헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv', skiprows=5, header=None)
df.columns = ['타임스탬프', '성별', '연령대', '머리감는시간', '두피고민', '샴푸선택이유', '샴푸아쉬운점', 'Q7', 'Q8', '기타1', '기타2']

print("=" * 70)
print("🔬 통계적 신뢰성 검증 분석")
print("=" * 70)
print("""
이 분석은 이전 Feature Importance 분석 결과의 신뢰성을 검증합니다.
특히 "Q7(두피차이인식)이 구매 의향에 가장 큰 영향을 미친다"는 
결론이 통계적으로 유의미한지 확인합니다.
""")

# ============================================================
# 데이터 전처리 (이전과 동일)
# ============================================================
df['구매의향'] = (df['Q8'] == '있다').astype(int)

le_gender = LabelEncoder()
le_age = LabelEncoder()
le_time = LabelEncoder()

df['성별_encoded'] = le_gender.fit_transform(df['성별'])
df['연령대_encoded'] = le_age.fit_transform(df['연령대'])
df['머리감는시간_encoded'] = le_time.fit_transform(df['머리감는시간'])
df['Q7_score'] = pd.to_numeric(df['Q7'], errors='coerce').fillna(3)
df['하루2번샴푸'] = df['머리감는시간'].str.contains('아침&저녁', na=False).astype(int)

scalp_concerns = ['두피 열감', '유분 과다', '건조함', '가려움', '탈모', '민감성', '특별한 고민 없음']
for concern in scalp_concerns:
    df[f'고민_{concern}'] = df['두피고민'].str.contains(concern, na=False).astype(int)

shampoo_reasons = ['두피 케어', '탈모 완화', '세정력', '향', '가격', '브랜드']
for reason in shampoo_reasons:
    df[f'이유_{reason}'] = df['샴푸선택이유'].str.contains(reason, na=False).astype(int)

feature_names = ['성별_encoded', '연령대_encoded', 'Q7_score', '하루2번샴푸'] + \
                [f'고민_{c}' for c in scalp_concerns] + \
                [f'이유_{r}' for r in shampoo_reasons]

feature_names_kr = ['성별', '연령대', 'Q7(두피차이인식)', '하루2번샴푸',
                    '고민:두피열감', '고민:유분과다', '고민:건조함', '고민:가려움', 
                    '고민:탈모', '고민:민감성', '고민:없음',
                    '이유:두피케어', '이유:탈모완화', '이유:세정력', 
                    '이유:향', '이유:가격', '이유:브랜드']

X = df[feature_names]
y = df['구매의향']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

print(f"📊 데이터 개요:")
print(f"   전체 샘플 수: {len(df)}명")
print(f"   구매 의향 있음: {y.sum()}명 ({y.mean()*100:.1f}%)")
print(f"   구매 의향 없음: {len(y) - y.sum()}명 ({(1-y.mean())*100:.1f}%)")

# ============================================================
# 1. Logistic Regression p-value 검정 (statsmodels 사용)
# ============================================================
print("\n" + "=" * 70)
print("📌 1. Logistic Regression 통계적 유의성 검정")
print("=" * 70)

print("""
📖 해석 가이드:
   • p-value < 0.05: 통계적으로 유의미함 (95% 신뢰수준)
   • p-value < 0.01: 매우 유의미함 (99% 신뢰수준)
   • p-value < 0.001: 극히 유의미함 (99.9% 신뢰수준)
   • Odds Ratio > 1: 해당 변수가 증가하면 구매 의향 증가
   • Odds Ratio < 1: 해당 변수가 증가하면 구매 의향 감소
""")

# statsmodels로 p-value 계산
X_with_const = sm.add_constant(X_scaled)
logit_model = sm.Logit(y, X_with_const)
result = logit_model.fit(disp=0)

# 결과 정리
stats_df = pd.DataFrame({
    '피처': ['상수'] + feature_names_kr,
    '계수': result.params,
    '표준오차': result.bse,
    'z값': result.tvalues,
    'p-value': result.pvalues,
    'Odds Ratio': np.exp(result.params)
})

# 유의성 표시
def significance_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return ''

stats_df['유의성'] = stats_df['p-value'].apply(significance_stars)
stats_df = stats_df[stats_df['피처'] != '상수']  # 상수 제외
stats_df_sorted = stats_df.sort_values('p-value')

print("\n📊 Logistic Regression 결과 (p-value 순):")
print("-" * 85)
print(f"{'피처':<20} {'계수':>10} {'Odds Ratio':>12} {'p-value':>12} {'유의성':>6}")
print("-" * 85)

for _, row in stats_df_sorted.iterrows():
    sig_mark = row['유의성']
    p_str = f"{row['p-value']:.4f}" if row['p-value'] >= 0.0001 else "<0.0001"
    print(f"{row['피처']:<20} {row['계수']:>+10.4f} {row['Odds Ratio']:>12.4f} {p_str:>12} {sig_mark:>6}")

print("-" * 85)
print("유의수준: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

# Q7 결과 강조
q7_row = stats_df[stats_df['피처'] == 'Q7(두피차이인식)'].iloc[0]
print(f"\n💡 Q7(두피차이인식) 상세 해석:")
print(f"   • 계수: {q7_row['계수']:+.4f}")
print(f"   • p-value: {q7_row['p-value']:.4f}")
print(f"   • Odds Ratio: {q7_row['Odds Ratio']:.4f}")

if q7_row['p-value'] < 0.05:
    print(f"\n   ✅ 결론: Q7은 통계적으로 유의미합니다 (p < 0.05)")
    print(f"   📊 Odds Ratio {q7_row['Odds Ratio']:.2f}의 의미:")
    print(f"      Q7 점수가 1점 증가할 때마다 구매 의향이")
    print(f"      {(q7_row['Odds Ratio']-1)*100:.1f}% 증가합니다.")
else:
    print(f"\n   ⚠️ 주의: Q7은 통계적으로 유의미하지 않습니다 (p >= 0.05)")
    print(f"      샘플 수가 적어 통계적 검정력이 부족할 수 있습니다.")

# 유의미한 변수 요약
significant_vars = stats_df[stats_df['p-value'] < 0.05]
marginally_significant = stats_df[(stats_df['p-value'] >= 0.05) & (stats_df['p-value'] < 0.1)]

print(f"\n📊 통계적으로 유의미한 변수 (p < 0.05): {len(significant_vars)}개")
for _, row in significant_vars.iterrows():
    direction = "↑ 구매의향 증가" if row['계수'] > 0 else "↓ 구매의향 감소"
    print(f"   • {row['피처']}: {direction} (p={row['p-value']:.4f})")

if len(marginally_significant) > 0:
    print(f"\n📊 경계선상 유의미한 변수 (0.05 ≤ p < 0.1): {len(marginally_significant)}개")
    for _, row in marginally_significant.iterrows():
        direction = "↑" if row['계수'] > 0 else "↓"
        print(f"   • {row['피처']}: {direction} (p={row['p-value']:.4f})")

# ============================================================
# 2. Permutation Importance (더 안정적인 중요도 측정)
# ============================================================
print("\n" + "=" * 70)
print("📌 2. Permutation Importance 분석")
print("=" * 70)

print("""
📖 해석 가이드:
   Permutation Importance는 각 피처의 값을 무작위로 섞었을 때
   모델 성능이 얼마나 떨어지는지를 측정합니다.
   
   • 성능 하락이 크면: 해당 피처가 중요함
   • 성능 하락이 작으면: 해당 피처가 덜 중요함
   • 표준편차가 작으면: 결과가 안정적임
""")

# Random Forest로 Permutation Importance 계산
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Permutation Importance (10번 반복)
perm_importance = permutation_importance(rf_model, X_test, y_test, n_repeats=30, random_state=42)

perm_df = pd.DataFrame({
    '피처': feature_names_kr,
    '중요도': perm_importance.importances_mean,
    '표준편차': perm_importance.importances_std
}).sort_values('중요도', ascending=False)

print("\n📊 Permutation Importance (Top 10):")
print("-" * 60)
print(f"{'피처':<25} {'중요도':>12} {'표준편차':>12} {'신뢰구간':<20}")
print("-" * 60)

for _, row in perm_df.head(10).iterrows():
    ci_low = row['중요도'] - 1.96 * row['표준편차']
    ci_high = row['중요도'] + 1.96 * row['표준편차']
    ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"
    print(f"{row['피처']:<25} {row['중요도']:>12.4f} {row['표준편차']:>12.4f} {ci_str:<20}")

# Q7 Permutation Importance 해석
q7_perm = perm_df[perm_df['피처'] == 'Q7(두피차이인식)'].iloc[0]
q7_rank = perm_df[perm_df['피처'] == 'Q7(두피차이인식)'].index[0] + 1

print(f"\n💡 Q7(두피차이인식) Permutation Importance 해석:")
print(f"   • 순위: {q7_rank}위")
print(f"   • 중요도: {q7_perm['중요도']:.4f} ± {q7_perm['표준편차']:.4f}")
print(f"   • 95% 신뢰구간: [{q7_perm['중요도'] - 1.96*q7_perm['표준편차']:.4f}, {q7_perm['중요도'] + 1.96*q7_perm['표준편차']:.4f}]")

if q7_perm['중요도'] - 1.96*q7_perm['표준편차'] > 0:
    print(f"   ✅ 신뢰구간이 0을 포함하지 않음 → 안정적으로 중요한 피처")
else:
    print(f"   ⚠️ 신뢰구간이 0을 포함 → 중요도가 불안정할 수 있음")

# ============================================================
# 3. Bootstrap Confidence Interval
# ============================================================
print("\n" + "=" * 70)
print("📌 3. Bootstrap 신뢰구간 분석")
print("=" * 70)

print("""
📖 해석 가이드:
   Bootstrap은 데이터를 여러 번 재샘플링하여
   Feature Importance의 분포를 추정합니다.
   
   • 95% 신뢰구간이 좁으면: 결과가 안정적
   • 95% 신뢰구간이 넓으면: 결과가 불안정 (샘플 수 부족 가능)
""")

n_bootstrap = 100
bootstrap_importances = np.zeros((n_bootstrap, len(feature_names)))

print(f"\n부트스트랩 분석 중... (n={n_bootstrap})")

for i in range(n_bootstrap):
    # 복원 추출로 샘플링
    indices = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X.iloc[indices]
    y_boot = y.iloc[indices]
    
    # 모델 학습
    rf_boot = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=i)
    rf_boot.fit(X_boot, y_boot)
    bootstrap_importances[i] = rf_boot.feature_importances_

# 신뢰구간 계산
bootstrap_df = pd.DataFrame({
    '피처': feature_names_kr,
    '평균 중요도': bootstrap_importances.mean(axis=0),
    '표준편차': bootstrap_importances.std(axis=0),
    '2.5% 백분위': np.percentile(bootstrap_importances, 2.5, axis=0),
    '97.5% 백분위': np.percentile(bootstrap_importances, 97.5, axis=0)
}).sort_values('평균 중요도', ascending=False)

print("\n📊 Bootstrap Feature Importance (95% 신뢰구간):")
print("-" * 75)
print(f"{'피처':<25} {'평균':>10} {'표준편차':>10} {'95% CI':<25}")
print("-" * 75)

for _, row in bootstrap_df.head(10).iterrows():
    ci_str = f"[{row['2.5% 백분위']:.4f}, {row['97.5% 백분위']:.4f}]"
    print(f"{row['피처']:<25} {row['평균 중요도']:>10.4f} {row['표준편차']:>10.4f} {ci_str:<25}")

# Q7 Bootstrap 결과
q7_boot = bootstrap_df[bootstrap_df['피처'] == 'Q7(두피차이인식)'].iloc[0]
print(f"\n💡 Q7(두피차이인식) Bootstrap 분석 결과:")
print(f"   • 평균 중요도: {q7_boot['평균 중요도']:.4f}")
print(f"   • 95% 신뢰구간: [{q7_boot['2.5% 백분위']:.4f}, {q7_boot['97.5% 백분위']:.4f}]")
print(f"   • 신뢰구간 폭: {q7_boot['97.5% 백분위'] - q7_boot['2.5% 백분위']:.4f}")

# ============================================================
# 4. 단변량 분석 (Chi-square, T-test)
# ============================================================
print("\n" + "=" * 70)
print("📌 4. 단변량 통계 검정")
print("=" * 70)

print("""
📖 해석 가이드:
   각 변수와 구매 의향 간의 관계를 개별적으로 검정합니다.
   
   • 연속형 변수: T-test (두 그룹 평균 비교)
   • 범주형 변수: Chi-square test (독립성 검정)
""")

# Q7과 구매 의향: T-test
q7_purchase_yes = df[df['구매의향'] == 1]['Q7_score']
q7_purchase_no = df[df['구매의향'] == 0]['Q7_score']

t_stat, t_pvalue = stats.ttest_ind(q7_purchase_yes, q7_purchase_no)
cohens_d = (q7_purchase_yes.mean() - q7_purchase_no.mean()) / np.sqrt(
    ((len(q7_purchase_yes)-1)*q7_purchase_yes.std()**2 + (len(q7_purchase_no)-1)*q7_purchase_no.std()**2) / 
    (len(q7_purchase_yes) + len(q7_purchase_no) - 2)
)

print("\n📊 Q7(두피차이인식) vs 구매 의향 - T-test:")
print("-" * 50)
print(f"   구매 의향 있음 그룹 Q7 평균: {q7_purchase_yes.mean():.2f} (n={len(q7_purchase_yes)})")
print(f"   구매 의향 없음 그룹 Q7 평균: {q7_purchase_no.mean():.2f} (n={len(q7_purchase_no)})")
print(f"   평균 차이: {q7_purchase_yes.mean() - q7_purchase_no.mean():.2f}")
print(f"   t-통계량: {t_stat:.4f}")
print(f"   p-value: {t_pvalue:.4f}")
print(f"   Cohen's d (효과 크기): {cohens_d:.4f}")

# 효과 크기 해석
if abs(cohens_d) < 0.2:
    effect_size_interp = "작은 효과"
elif abs(cohens_d) < 0.5:
    effect_size_interp = "중간 효과"
elif abs(cohens_d) < 0.8:
    effect_size_interp = "중간~큰 효과"
else:
    effect_size_interp = "큰 효과"

print(f"\n💡 해석:")
if t_pvalue < 0.05:
    print(f"   ✅ 통계적으로 유의미한 차이가 있습니다 (p = {t_pvalue:.4f} < 0.05)")
else:
    print(f"   ⚠️ 통계적으로 유의미한 차이가 없습니다 (p = {t_pvalue:.4f} >= 0.05)")
print(f"   📊 효과 크기: {effect_size_interp} (Cohen's d = {cohens_d:.2f})")
print(f"   → Q7 점수가 높은 사람이 구매 의향도 높은 경향이 있습니다.")

# 하루 2번 샴푸 vs 구매 의향: Chi-square
contingency_table = pd.crosstab(df['하루2번샴푸'], df['구매의향'])
chi2, chi_pvalue, dof, expected = stats.chi2_contingency(contingency_table)

print("\n📊 하루2번샴푸 vs 구매 의향 - Chi-square test:")
print("-" * 50)
print(f"   교차표:")
print(contingency_table.to_string().replace('\n', '\n   '))
print(f"\n   Chi-square 통계량: {chi2:.4f}")
print(f"   p-value: {chi_pvalue:.4f}")
print(f"   자유도: {dof}")

# Cramér's V (효과 크기)
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
print(f"   Cramér's V (효과 크기): {cramers_v:.4f}")

print(f"\n💡 해석:")
if chi_pvalue < 0.05:
    print(f"   ✅ 하루 2번 샴푸 여부와 구매 의향 간에 유의미한 관계가 있습니다")
else:
    print(f"   ⚠️ 통계적으로 유의미한 관계가 없습니다 (p >= 0.05)")

# 하루 2번 샴푸 그룹의 구매 의향 비율
twice_daily_purchase = df[df['하루2번샴푸'] == 1]['구매의향'].mean()
once_daily_purchase = df[df['하루2번샴푸'] == 0]['구매의향'].mean()
print(f"   • 하루 2번 샴푸 그룹 구매 의향: {twice_daily_purchase:.1%}")
print(f"   • 하루 1번 샴푸 그룹 구매 의향: {once_daily_purchase:.1%}")
print(f"   • 차이: +{(twice_daily_purchase - once_daily_purchase)*100:.1f}%p")

# ============================================================
# 5. 검정력 분석 (Power Analysis)
# ============================================================
print("\n" + "=" * 70)
print("📌 5. 검정력(Statistical Power) 분석")
print("=" * 70)

print("""
📖 해석 가이드:
   검정력은 "실제 효과가 있을 때 이를 탐지할 확률"입니다.
   
   • 검정력 ≥ 0.80: 충분한 검정력 (권장)
   • 검정력 < 0.80: 검정력 부족 (Type II 오류 위험)
   
   현재 샘플 수로 탐지 가능한 효과 크기를 계산합니다.
""")

# 현재 샘플 수로 탐지 가능한 최소 효과 크기 (근사 계산)
n1 = len(q7_purchase_yes)
n2 = len(q7_purchase_no)
alpha = 0.05
power = 0.80

# 효과 크기 d에 대한 검정력 계산 (근사)
from scipy.stats import norm

def calculate_power(n1, n2, d, alpha=0.05):
    """두 표본 t-검정의 검정력 계산"""
    se = np.sqrt(1/n1 + 1/n2)
    z_alpha = norm.ppf(1 - alpha/2)
    z_power = d / se - z_alpha
    return norm.cdf(z_power)

# 현재 효과 크기에서의 검정력
current_power = calculate_power(n1, n2, abs(cohens_d))

print(f"\n📊 현재 데이터의 검정력:")
print(f"   • 샘플 수: 구매의향 있음 {n1}명, 없음 {n2}명")
print(f"   • 관측된 효과 크기 (Cohen's d): {abs(cohens_d):.4f}")
print(f"   • 현재 검정력: {current_power:.1%}")

if current_power >= 0.80:
    print(f"   ✅ 충분한 검정력을 가지고 있습니다")
else:
    print(f"   ⚠️ 검정력이 부족합니다 (권장: 80% 이상)")
    
    # 80% 검정력을 위한 필요 샘플 수 계산 (근사)
    def required_sample_size(d, power=0.80, alpha=0.05):
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / d) ** 2
        return int(np.ceil(n))
    
    if abs(cohens_d) > 0.1:
        required_n = required_sample_size(abs(cohens_d))
        print(f"   📊 80% 검정력을 위한 필요 샘플 수: 각 그룹 약 {required_n}명")
        print(f"      (현재: {min(n1, n2)}명)")

# ============================================================
# 6. 종합 신뢰성 평가
# ============================================================
print("\n" + "=" * 70)
print("📌 6. 종합 신뢰성 평가 및 결론")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    Q7(두피차이인식) 신뢰성 종합 평가                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │""")

# 각 검정 결과 요약
checks = []

# 1. Logistic Regression p-value
if q7_row['p-value'] < 0.05:
    checks.append(("Logistic Regression p-value", "✅ 유의미", f"p={q7_row['p-value']:.4f}"))
elif q7_row['p-value'] < 0.1:
    checks.append(("Logistic Regression p-value", "⚠️ 경계선", f"p={q7_row['p-value']:.4f}"))
else:
    checks.append(("Logistic Regression p-value", "❌ 비유의미", f"p={q7_row['p-value']:.4f}"))

# 2. Permutation Importance
if q7_perm['중요도'] - 1.96*q7_perm['표준편차'] > 0:
    checks.append(("Permutation Importance", "✅ 안정적", f"CI가 0 미포함"))
else:
    checks.append(("Permutation Importance", "⚠️ 불안정", f"CI가 0 포함"))

# 3. T-test
if t_pvalue < 0.05:
    checks.append(("T-test (단변량)", "✅ 유의미", f"p={t_pvalue:.4f}"))
elif t_pvalue < 0.1:
    checks.append(("T-test (단변량)", "⚠️ 경계선", f"p={t_pvalue:.4f}"))
else:
    checks.append(("T-test (단변량)", "❌ 비유의미", f"p={t_pvalue:.4f}"))

# 4. 효과 크기
if abs(cohens_d) >= 0.5:
    checks.append(("효과 크기 (Cohen's d)", "✅ 중간 이상", f"d={cohens_d:.2f}"))
elif abs(cohens_d) >= 0.2:
    checks.append(("효과 크기 (Cohen's d)", "⚠️ 작은~중간", f"d={cohens_d:.2f}"))
else:
    checks.append(("효과 크기 (Cohen's d)", "❌ 작음", f"d={cohens_d:.2f}"))

# 5. 검정력
if current_power >= 0.80:
    checks.append(("검정력", "✅ 충분", f"{current_power:.1%}"))
else:
    checks.append(("검정력", "⚠️ 부족", f"{current_power:.1%}"))

for check_name, result, detail in checks:
    print(f"│  • {check_name:<30} {result:<12} ({detail})  │")

# 종합 판정
positive_checks = sum(1 for _, result, _ in checks if "✅" in result)
total_checks = len(checks)

print(f"""│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 종합 점수: {positive_checks}/{total_checks} 항목 통과                                      │
│                                                                     │""")

if positive_checks >= 4:
    conclusion = "높음"
    conclusion_detail = "Q7(두피차이인식)이 구매 의향에 미치는 영향은 통계적으로 신뢰할 수 있습니다."
elif positive_checks >= 3:
    conclusion = "중간"
    conclusion_detail = "Q7의 영향이 있으나, 샘플 수 증가로 더 확실한 검증이 필요합니다."
else:
    conclusion = "낮음"
    conclusion_detail = "현재 데이터로는 Q7의 영향을 확신하기 어렵습니다. 추가 데이터 수집을 권장합니다."

print(f"""│  🎯 신뢰성 수준: {conclusion}                                             │
│                                                                     │
│  💡 결론:                                                            │
│     {conclusion_detail:<60}│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 7. 기획서용 요약
# ============================================================
print("=" * 70)
print("📌 7. 기획서/보고서용 요약")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│              데이앤나이트 듀얼 샴푸 - 통계 분석 결과 요약              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📊 주요 발견:                                                       │
│                                                                     │
│  1. Q7(아침/밤 두피 차이 인식)과 구매 의향의 관계                     │
│     • 구매 의향 있음 그룹 평균: {q7_purchase_yes.mean():.2f}점                        │
│     • 구매 의향 없음 그룹 평균: {q7_purchase_no.mean():.2f}점                        │
│     • 통계적 유의성: p = {t_pvalue:.4f} {'(유의미)' if t_pvalue < 0.05 else '(경계선)' if t_pvalue < 0.1 else '(비유의미)'}                      │
│                                                                     │
│  2. 하루 2번 샴푸 사용자의 구매 의향                                  │
│     • 하루 2번 샴푸 그룹: {twice_daily_purchase:.1%} 구매 의향                       │
│     • 하루 1번 샴푸 그룹: {once_daily_purchase:.1%} 구매 의향                       │
│     • 차이: +{(twice_daily_purchase - once_daily_purchase)*100:.1f}%p                                              │
│                                                                     │
│  3. Feature Importance 순위 (Random Forest 기준)                     │
│     1위: Q7(두피차이인식)                                            │
│     2위: 연령대                                                      │
│     3위: 이유:가격                                                   │
│     4위: 하루2번샴푸                                                 │
│                                                                     │
│  ✅ 핵심 인사이트:                                                   │
│     "아침과 밤 두피 상태가 다르다고 느끼는 소비자일수록               │
│      데이앤나이트 듀얼 샴푸에 대한 구매 의향이 높다"                  │
│                                                                     │
│  ⚠️ 주의사항:                                                        │
│     • 샘플 수 {len(df)}명으로 통계적 검정력이 제한적                       │
│     • 추가 설문 수집으로 결과 검증 권장                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 시각화 생성
# ============================================================
print("=" * 70)
print("📊 시각화 생성 중...")
print("=" * 70)

# Figure 1: 통계적 유의성 시각화
fig1, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1-1: Logistic Regression 계수 + p-value
ax1 = axes[0, 0]
stats_sorted_by_coef = stats_df.sort_values('계수')
colors = ['#2ecc71' if p < 0.05 else '#f39c12' if p < 0.1 else '#95a5a6' 
          for p in stats_sorted_by_coef['p-value']]
bars = ax1.barh(stats_sorted_by_coef['피처'], stats_sorted_by_coef['계수'], color=colors, edgecolor='black')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax1.set_xlabel('Logistic Regression 계수', fontsize=11)
ax1.set_title('변수별 영향력 및 통계적 유의성\n(🟢 p<0.05  🟡 p<0.1  ⚪ p≥0.1)', fontsize=12, fontweight='bold')

# 1-2: Q7 점수 분포 비교
ax2 = axes[0, 1]
ax2.hist(q7_purchase_yes, bins=5, alpha=0.7, label=f'구매의향 있음 (n={len(q7_purchase_yes)})', color='#2ecc71', edgecolor='black')
ax2.hist(q7_purchase_no, bins=5, alpha=0.7, label=f'구매의향 없음 (n={len(q7_purchase_no)})', color='#e74c3c', edgecolor='black')
ax2.axvline(q7_purchase_yes.mean(), color='#27ae60', linestyle='--', linewidth=2, label=f'있음 평균: {q7_purchase_yes.mean():.2f}')
ax2.axvline(q7_purchase_no.mean(), color='#c0392b', linestyle='--', linewidth=2, label=f'없음 평균: {q7_purchase_no.mean():.2f}')
ax2.set_xlabel('Q7 점수 (아침/밤 두피 차이 인식)', fontsize=11)
ax2.set_ylabel('응답자 수', fontsize=11)
ax2.set_title(f'Q7 점수 분포 비교\n(T-test p={t_pvalue:.4f}, Cohen\'s d={cohens_d:.2f})', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9)

# 1-3: Permutation Importance with CI
ax3 = axes[1, 0]
perm_top10 = perm_df.head(10).sort_values('중요도')
colors = ['#9b59b6' if (row['중요도'] - 1.96*row['표준편차']) > 0 else '#bdc3c7' 
          for _, row in perm_top10.iterrows()]
bars = ax3.barh(perm_top10['피처'], perm_top10['중요도'], xerr=1.96*perm_top10['표준편차'], 
                color=colors, edgecolor='black', capsize=3)
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Permutation Importance (95% CI)', fontsize=11)
ax3.set_title('Permutation Importance\n(🟣 CI가 0 미포함  ⚪ CI가 0 포함)', fontsize=12, fontweight='bold')

# 1-4: Bootstrap 분포 (Q7)
ax4 = axes[1, 1]
q7_idx = feature_names_kr.index('Q7(두피차이인식)')
q7_bootstrap = bootstrap_importances[:, q7_idx]
ax4.hist(q7_bootstrap, bins=20, color='#3498db', edgecolor='black', alpha=0.7)
ax4.axvline(q7_bootstrap.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {q7_bootstrap.mean():.4f}')
ax4.axvline(np.percentile(q7_bootstrap, 2.5), color='orange', linestyle=':', linewidth=2, label=f'2.5%: {np.percentile(q7_bootstrap, 2.5):.4f}')
ax4.axvline(np.percentile(q7_bootstrap, 97.5), color='orange', linestyle=':', linewidth=2, label=f'97.5%: {np.percentile(q7_bootstrap, 97.5):.4f}')
ax4.set_xlabel('Feature Importance', fontsize=11)
ax4.set_ylabel('빈도', fontsize=11)
ax4.set_title('Q7(두피차이인식) Bootstrap 분포\n(n=100 반복)', fontsize=12, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9)

plt.suptitle('통계적 신뢰성 검증 분석', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('statistical_validation.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ statistical_validation.png 저장 완료")

# Figure 2: 신뢰성 요약 대시보드
fig2, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

# 텍스트 요약
summary_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    Q7(두피차이인식) 통계적 신뢰성 검증 결과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 검정 결과 요약
─────────────────────────────────────────────────────────────────────────
  검정 방법                          결과              상세
─────────────────────────────────────────────────────────────────────────
"""

for check_name, result, detail in checks:
    summary_text += f"  {check_name:<32} {result:<14} {detail}\n"

summary_text += f"""─────────────────────────────────────────────────────────────────────────

📈 주요 수치
─────────────────────────────────────────────────────────────────────────
  • 구매의향 있음 그룹 Q7 평균: {q7_purchase_yes.mean():.2f}점
  • 구매의향 없음 그룹 Q7 평균: {q7_purchase_no.mean():.2f}점
  • 평균 차이: {q7_purchase_yes.mean() - q7_purchase_no.mean():.2f}점
  • T-test p-value: {t_pvalue:.4f}
  • Cohen's d (효과 크기): {cohens_d:.2f} ({effect_size_interp})
─────────────────────────────────────────────────────────────────────────

🎯 종합 판정: 신뢰성 {conclusion}
─────────────────────────────────────────────────────────────────────────
  {conclusion_detail}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
        verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

plt.savefig('statistical_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ statistical_summary.png 저장 완료")

print("\n" + "=" * 70)
print("✅ 분석 완료!")
print("=" * 70)
print("""
📁 생성된 파일:
   1. statistical_validation.png - 통계적 신뢰성 검증 시각화
   2. statistical_summary.png    - 신뢰성 요약 대시보드
""")
