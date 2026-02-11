# 구매 의향 예측 모델 - 성별, 연령대, 두피 고민, 머리 감는 시간 등을 기반으로 Q8 구매 의향 예측
# 클러스터링 분석 - 응답자들을 유사한 특성으로 그룹화
# 연관 규칙 분석 - 두피 고민과 샴푸 선택 이유 간의 연관성 분석


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# CSV 파일 읽기
df = pd.read_csv('헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv', skiprows=5, header=None)
df.columns = ['타임스탬프', '성별', '연령대', '머리감는시간', '두피고민', '샴푸선택이유', '샴푸아쉬운점', 'Q7', 'Q8', '기타']

print("=" * 60)
print("📊 설문조사 ML 분석 리포트")
print("=" * 60)
print(f"\n총 응답자 수: {len(df)}명")
print(f"남성: {len(df[df['성별'] == '남성'])}명, 여성: {len(df[df['성별'] == '여성'])}명")

# ============================================================
# 1. 구매 의향 예측 모델 (Random Forest)
# ============================================================
print("\n" + "=" * 60)
print("🤖 1. 구매 의향 예측 모델 (Random Forest Classifier)")
print("=" * 60)

# 피처 인코딩
le_gender = LabelEncoder()
le_age = LabelEncoder()
le_time = LabelEncoder()
le_q8 = LabelEncoder()

df_ml = df.copy()
df_ml['성별_encoded'] = le_gender.fit_transform(df_ml['성별'])
df_ml['연령대_encoded'] = le_age.fit_transform(df_ml['연령대'])
df_ml['머리감는시간_encoded'] = le_time.fit_transform(df_ml['머리감는시간'])
df_ml['Q7_encoded'] = pd.to_numeric(df_ml['Q7'], errors='coerce').fillna(3)
df_ml['Q8_encoded'] = le_q8.fit_transform(df_ml['Q8'])

# 두피 고민 복수 응답 처리 (One-hot encoding)
scalp_concerns = ['두피 열감', '유분 과다 / 떡짐', '건조함 / 각질', '가려움', 
                  '탈모 / 모발 가늘어짐', '민감성 / 자극감', '특별한 고민 없음']

for concern in scalp_concerns:
    df_ml[f'고민_{concern}'] = df_ml['두피고민'].str.contains(concern, na=False).astype(int)

# 피처 선택
feature_cols = ['성별_encoded', '연령대_encoded', '머리감는시간_encoded', 'Q7_encoded'] + \
               [f'고민_{c}' for c in scalp_concerns]

X = df_ml[feature_cols]
y = df_ml['Q8_encoded']

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 모델 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train, y_train)

# 예측 및 평가
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n모델 정확도: {accuracy:.2%}")
print(f"\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=le_q8.classes_))

# 피처 중요도
print("\n📌 피처 중요도 (구매 의향에 영향을 미치는 요인):")
feature_importance = pd.DataFrame({
    '피처': feature_cols,
    '중요도': rf_model.feature_importances_
}).sort_values('중요도', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  - {row['피처']}: {row['중요도']:.3f}")

# ============================================================
# 2. 고객 세그먼테이션 (K-Means Clustering)
# ============================================================
print("\n" + "=" * 60)
print("👥 2. 고객 세그먼테이션 (K-Means Clustering)")
print("=" * 60)

# 클러스터링용 피처
cluster_features = ['성별_encoded', '연령대_encoded', '머리감는시간_encoded', 'Q7_encoded'] + \
                   [f'고민_{c}' for c in scalp_concerns]

X_cluster = df_ml[cluster_features].values

# K-Means 클러스터링 (3개 그룹)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_ml['클러스터'] = kmeans.fit_predict(X_cluster)

print("\n📊 클러스터별 특성 분석:")
for cluster_id in range(3):
    cluster_data = df_ml[df_ml['클러스터'] == cluster_id]
    print(f"\n[클러스터 {cluster_id + 1}] - {len(cluster_data)}명 ({len(cluster_data)/len(df_ml)*100:.1f}%)")
    
    # 성별 분포
    gender_dist = cluster_data['성별'].value_counts()
    print(f"  성별: {dict(gender_dist)}")
    
    # 연령대 분포
    age_dist = cluster_data['연령대'].value_counts().head(3)
    print(f"  주요 연령대: {dict(age_dist)}")
    
    # 구매 의향
    purchase_rate = (cluster_data['Q8'] == '있다').mean()
    print(f"  구매 의향 '있다' 비율: {purchase_rate:.1%}")
    
    # 주요 두피 고민
    top_concerns = []
    for concern in scalp_concerns:
        if cluster_data[f'고민_{concern}'].mean() > 0.3:
            top_concerns.append(concern)
    if top_concerns:
        print(f"  주요 두피 고민: {', '.join(top_concerns)}")

# ============================================================
# 3. 연관 분석 (두피 고민 ↔ 샴푸 선택 이유)
# ============================================================
print("\n" + "=" * 60)
print("🔗 3. 연관 분석 (두피 고민 → 샴푸 선택 이유)")
print("=" * 60)

# 주요 샴푸 선택 이유
shampoo_reasons = ['두피 케어 효과를 기대해서', '탈모 완화 기능이 있어서', '세정력이 좋아서',
                   '향이 마음에 들어서', '가격이 합리적이어서', '브랜드 신뢰도 / 인지도']

print("\n📌 두피 고민별 선호하는 샴푸 선택 이유:")
for concern in scalp_concerns[:6]:  # '특별한 고민 없음' 제외
    concern_users = df_ml[df_ml[f'고민_{concern}'] == 1]
    if len(concern_users) >= 5:  # 최소 5명 이상인 경우만
        print(f"\n[{concern}] ({len(concern_users)}명)")
        for reason in shampoo_reasons:
            count = concern_users['샴푸선택이유'].str.contains(reason, na=False).sum()
            if count > 0:
                pct = count / len(concern_users) * 100
                print(f"  - {reason}: {count}명 ({pct:.1f}%)")

# ============================================================
# 4. 교차 분석 (성별 × 연령대 × 구매 의향)
# ============================================================
print("\n" + "=" * 60)
print("📈 4. 교차 분석 (성별 × 연령대 × 구매 의향)")
print("=" * 60)

cross_tab = pd.crosstab([df_ml['성별'], df_ml['연령대']], df_ml['Q8'], margins=True)
print("\n성별 × 연령대별 구매 의향 분포:")
print(cross_tab)

# 구매 의향 비율 계산
print("\n📌 성별 × 연령대별 구매 의향 '있다' 비율:")
for gender in ['남성', '여성']:
    print(f"\n[{gender}]")
    gender_data = df_ml[df_ml['성별'] == gender]
    for age in sorted(gender_data['연령대'].unique()):
        age_data = gender_data[gender_data['연령대'] == age]
        if len(age_data) >= 3:  # 최소 3명 이상
            rate = (age_data['Q8'] == '있다').mean()
            print(f"  {age}: {rate:.1%} ({len(age_data)}명 중 {(age_data['Q8'] == '있다').sum()}명)")

# ============================================================
# 5. 핵심 인사이트 요약
# ============================================================
print("\n" + "=" * 60)
print("💡 5. 핵심 인사이트 요약")
print("=" * 60)

# 전체 구매 의향
total_purchase_rate = (df_ml['Q8'] == '있다').mean()
male_purchase_rate = (df_ml[df_ml['성별'] == '남성']['Q8'] == '있다').mean()
female_purchase_rate = (df_ml[df_ml['성별'] == '여성']['Q8'] == '있다').mean()

print(f"\n1️⃣ 전체 구매 의향 비율: {total_purchase_rate:.1%}")
print(f"   - 남성: {male_purchase_rate:.1%}")
print(f"   - 여성: {female_purchase_rate:.1%}")

# Q7 점수와 구매 의향 관계
q7_purchase_corr = df_ml['Q7_encoded'].corr(df_ml['Q8_encoded'])
print(f"\n2️⃣ 두피 상태 변화 인식(Q7)과 구매 의향 상관관계: {q7_purchase_corr:.3f}")

# 가장 구매 의향이 높은 두피 고민
print(f"\n3️⃣ 두피 고민별 구매 의향 '있다' 비율:")
for concern in scalp_concerns:
    concern_users = df_ml[df_ml[f'고민_{concern}'] == 1]
    if len(concern_users) >= 5:
        rate = (concern_users['Q8'] == '있다').mean()
        print(f"   - {concern}: {rate:.1%} ({len(concern_users)}명)")

print("\n" + "=" * 60)
print("분석 완료!")
print("=" * 60)
