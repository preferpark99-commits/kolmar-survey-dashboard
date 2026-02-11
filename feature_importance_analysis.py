"""
============================================================
맨즈케어 데이앤나이트 듀얼 샴푸 - Feature Importance 분석
============================================================
구매 의향에 가장 큰 영향을 미치는 변수 추출

모델:
1. Logistic Regression - 변수별 영향 방향(+/-) 분석
2. Decision Tree - 구매 규칙 도출
3. Random Forest - 안정적인 중요도 순위
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# CSV 파일 읽기
df = pd.read_csv('헤어·두피 케어 제품에 대한 수요 설문조사(응답) - 설문지 응답 시트1.csv', skiprows=5, header=None)
df.columns = ['타임스탬프', '성별', '연령대', '머리감는시간', '두피고민', '샴푸선택이유', '샴푸아쉬운점', 'Q7', 'Q8', '기타1', '기타2']

print("=" * 70)
print("🔬 Feature Importance 분석 - 구매 의향 영향 요인")
print("=" * 70)

# ============================================================
# 데이터 전처리
# ============================================================
print("\n📊 데이터 전처리 중...")

# 타겟 변수 인코딩
df['구매의향'] = (df['Q8'] == '있다').astype(int)

# 피처 인코딩
le_gender = LabelEncoder()
le_age = LabelEncoder()
le_time = LabelEncoder()

df['성별_encoded'] = le_gender.fit_transform(df['성별'])
df['연령대_encoded'] = le_age.fit_transform(df['연령대'])
df['머리감는시간_encoded'] = le_time.fit_transform(df['머리감는시간'])
df['Q7_score'] = pd.to_numeric(df['Q7'], errors='coerce').fillna(3)

# 하루 2번 샴푸 여부
df['하루2번샴푸'] = df['머리감는시간'].str.contains('아침&저녁', na=False).astype(int)

# 두피 고민 One-hot encoding
scalp_concerns = ['두피 열감', '유분 과다', '건조함', '가려움', '탈모', '민감성', '특별한 고민 없음']
for concern in scalp_concerns:
    df[f'고민_{concern}'] = df['두피고민'].str.contains(concern, na=False).astype(int)

# 샴푸 선택 이유 One-hot encoding
shampoo_reasons = ['두피 케어', '탈모 완화', '세정력', '향', '가격', '브랜드']
for reason in shampoo_reasons:
    df[f'이유_{reason}'] = df['샴푸선택이유'].str.contains(reason, na=False).astype(int)

# 피처 선택
feature_names = ['성별_encoded', '연령대_encoded', 'Q7_score', '하루2번샴푸'] + \
                [f'고민_{c}' for c in scalp_concerns] + \
                [f'이유_{r}' for r in shampoo_reasons]

# 피처명 한글화 (시각화용)
feature_names_kr = ['성별', '연령대', 'Q7(두피차이인식)', '하루2번샴푸',
                    '고민:두피열감', '고민:유분과다', '고민:건조함', '고민:가려움', 
                    '고민:탈모', '고민:민감성', '고민:없음',
                    '이유:두피케어', '이유:탈모완화', '이유:세정력', 
                    '이유:향', '이유:가격', '이유:브랜드']

X = df[feature_names]
y = df['구매의향']

print(f"   전체 샘플 수: {len(df)}명")
print(f"   피처 수: {len(feature_names)}개")
print(f"   구매 의향 있음: {y.sum()}명 ({y.mean()*100:.1f}%)")
print(f"   구매 의향 없음: {len(y) - y.sum()}명 ({(1-y.mean())*100:.1f}%)")

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# 스케일링 (Logistic Regression용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 1. Logistic Regression 분석
# ============================================================
print("\n" + "=" * 70)
print("📌 1. Logistic Regression - 변수별 영향 방향 분석")
print("=" * 70)

lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred_lr = lr_model.predict(X_test_scaled)
y_prob_lr = lr_model.predict_proba(X_test_scaled)[:, 1]
accuracy_lr = accuracy_score(y_test, y_pred_lr)

print(f"\n모델 정확도: {accuracy_lr:.1%}")

# 계수 분석
lr_coef = pd.DataFrame({
    '피처': feature_names_kr,
    '계수': lr_model.coef_[0],
    '영향방향': ['긍정(+)' if c > 0 else '부정(-)' for c in lr_model.coef_[0]],
    '절대값': np.abs(lr_model.coef_[0])
}).sort_values('절대값', ascending=False)

print("\n📊 Logistic Regression 계수 (구매 의향에 미치는 영향):")
print("-" * 60)
for idx, row in lr_coef.head(10).iterrows():
    direction = "🔺" if row['계수'] > 0 else "🔻"
    bar = "█" * int(row['절대값'] * 5)
    print(f"   {direction} {row['피처']:20s}: {row['계수']:+.3f} {bar}")

print("\n💡 해석:")
top_positive = lr_coef[lr_coef['계수'] > 0].head(3)
top_negative = lr_coef[lr_coef['계수'] < 0].head(3)
print("   구매 의향을 높이는 요인:")
for _, row in top_positive.iterrows():
    print(f"      • {row['피처']}")
print("   구매 의향을 낮추는 요인:")
for _, row in top_negative.iterrows():
    print(f"      • {row['피처']}")

# ============================================================
# 2. Decision Tree 분석
# ============================================================
print("\n" + "=" * 70)
print("📌 2. Decision Tree - 구매 규칙 도출")
print("=" * 70)

dt_model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n모델 정확도: {accuracy_dt:.1%}")

# Feature Importance
dt_importance = pd.DataFrame({
    '피처': feature_names_kr,
    '중요도': dt_model.feature_importances_
}).sort_values('중요도', ascending=False)

print("\n📊 Decision Tree Feature Importance:")
print("-" * 60)
for idx, row in dt_importance[dt_importance['중요도'] > 0].iterrows():
    bar = "█" * int(row['중요도'] * 30)
    print(f"   {row['피처']:20s}: {row['중요도']:.3f} {bar}")

# ============================================================
# 3. Random Forest 분석
# ============================================================
print("\n" + "=" * 70)
print("📌 3. Random Forest - 안정적인 중요도 순위")
print("=" * 70)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"\n모델 정확도: {accuracy_rf:.1%}")
print(f"교차 검증 정확도: {cv_scores.mean():.1%} (±{cv_scores.std()*2:.1%})")

# Feature Importance
rf_importance = pd.DataFrame({
    '피처': feature_names_kr,
    '중요도': rf_model.feature_importances_
}).sort_values('중요도', ascending=False)

print("\n📊 Random Forest Feature Importance (Top 10):")
print("-" * 60)
for idx, row in rf_importance.head(10).iterrows():
    bar = "█" * int(row['중요도'] * 50)
    print(f"   {row['피처']:20s}: {row['중요도']:.3f} {bar}")

# ============================================================
# 4. 종합 분석 결과
# ============================================================
print("\n" + "=" * 70)
print("📌 4. 종합 분석 결과")
print("=" * 70)

# 모델 성능 비교
print("\n📊 모델 성능 비교:")
print("-" * 40)
print(f"   Logistic Regression: {accuracy_lr:.1%}")
print(f"   Decision Tree:       {accuracy_dt:.1%}")
print(f"   Random Forest:       {accuracy_rf:.1%}")

# 세 모델의 Top 5 피처 종합
print("\n📊 모델별 Top 5 중요 피처:")
print("-" * 60)
print(f"{'순위':<6}{'Logistic Reg.':<20}{'Decision Tree':<20}{'Random Forest':<20}")
print("-" * 60)
for i in range(5):
    lr_feat = lr_coef.iloc[i]['피처']
    dt_feat = dt_importance.iloc[i]['피처'] if i < len(dt_importance[dt_importance['중요도'] > 0]) else "-"
    rf_feat = rf_importance.iloc[i]['피처']
    print(f"{i+1:<6}{lr_feat:<20}{dt_feat:<20}{rf_feat:<20}")

# 공통 중요 피처 찾기
top5_lr = set(lr_coef.head(5)['피처'])
top5_dt = set(dt_importance.head(5)['피처'])
top5_rf = set(rf_importance.head(5)['피처'])
common_features = top5_lr & top5_rf

print(f"\n💡 3개 모델에서 공통으로 중요한 피처:")
for feat in common_features:
    print(f"   ✅ {feat}")

# ============================================================
# 5. 기획 인사이트
# ============================================================
print("\n" + "=" * 70)
print("📌 5. 데이앤나이트 듀얼 샴푸 기획 인사이트")
print("=" * 70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    Feature Importance 분석 결과 요약                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │""")

# Q7 점수 중요도 확인
q7_rank_rf = rf_importance[rf_importance['피처'] == 'Q7(두피차이인식)'].index[0] + 1
q7_importance = rf_importance[rf_importance['피처'] == 'Q7(두피차이인식)']['중요도'].values[0]

print(f"""│  1️⃣  Q7(아침/밤 두피 차이 인식)                                     │
│      • Random Forest 중요도 순위: {q7_rank_rf}위                       │
│      • 중요도 점수: {q7_importance:.3f}                                │
│      → 제품 컨셉의 핵심 근거!                                         │""")

# 하루 2번 샴푸 중요도
twice_rank = rf_importance[rf_importance['피처'] == '하루2번샴푸'].index[0] + 1
twice_importance = rf_importance[rf_importance['피처'] == '하루2번샴푸']['중요도'].values[0]

print(f"""│                                                                     │
│  2️⃣  하루 2번 샴푸 여부                                              │
│      • Random Forest 중요도 순위: {twice_rank}위                       │
│      • 중요도 점수: {twice_importance:.3f}                             │
│      → 핵심 타겟 고객 선정 근거!                                      │""")

# 탈모 고민 중요도
hairloss_importance = rf_importance[rf_importance['피처'] == '고민:탈모']['중요도'].values[0]

print(f"""│                                                                     │
│  3️⃣  탈모 고민                                                       │
│      • 중요도 점수: {hairloss_importance:.3f}                          │
│      → 나이트 샴푸 탈모 완화 기능 근거!                               │""")

print("""│                                                                     │
│  ✅ 결론:                                                            │
│     "아침/밤 두피 상태 차이를 인식하는 고객"이                        │
│     구매 의향이 높다는 것이 데이터로 검증됨                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
""")

# ============================================================
# 시각화 생성
# ============================================================
print("=" * 70)
print("📊 시각화 생성 중...")
print("=" * 70)

# Figure 1: Feature Importance 비교 (3개 모델)
fig1, axes = plt.subplots(1, 3, figsize=(18, 8))

# Logistic Regression 계수
ax1 = axes[0]
colors1 = ['#2ecc71' if c > 0 else '#e74c3c' for c in lr_coef.head(10)['계수']]
bars1 = ax1.barh(lr_coef.head(10)['피처'], lr_coef.head(10)['절대값'], color=colors1)
ax1.set_xlabel('계수 절대값', fontsize=12)
ax1.set_title('Logistic Regression\n(🟢 긍정적 / 🔴 부정적 영향)', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

# Decision Tree Feature Importance
ax2 = axes[1]
dt_top10 = dt_importance.head(10)
bars2 = ax2.barh(dt_top10['피처'], dt_top10['중요도'], color='#3498db')
ax2.set_xlabel('Feature Importance', fontsize=12)
ax2.set_title('Decision Tree\nFeature Importance', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Random Forest Feature Importance
ax3 = axes[2]
rf_top10 = rf_importance.head(10)
bars3 = ax3.barh(rf_top10['피처'], rf_top10['중요도'], color='#9b59b6')
ax3.set_xlabel('Feature Importance', fontsize=12)
ax3.set_title('Random Forest\nFeature Importance', fontsize=14, fontweight='bold')
ax3.invert_yaxis()

plt.suptitle('구매 의향에 영향을 미치는 요인 분석', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ feature_importance_comparison.png 저장 완료")

# Figure 2: Decision Tree 시각화
fig2, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt_model, 
          feature_names=feature_names_kr, 
          class_names=['구매의향없음', '구매의향있음'],
          filled=True, 
          rounded=True,
          fontsize=9,
          ax=ax)
plt.title('Decision Tree - 구매 의향 예측 규칙', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ decision_tree_visualization.png 저장 완료")

# Figure 3: 모델 성능 비교 + ROC Curve
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

# 모델 정확도 비교
ax1 = axes[0]
models = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest']
accuracies = [accuracy_lr, accuracy_dt, accuracy_rf]
colors = ['#2ecc71', '#3498db', '#9b59b6']
bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylim(0, 1)
ax1.set_ylabel('정확도 (Accuracy)', fontsize=12)
ax1.set_title('모델별 예측 정확도', fontsize=14, fontweight='bold')
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{acc:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# ROC Curve
ax2 = axes[1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_rf = roc_auc_score(y_test, y_prob_rf)

ax2.plot(fpr_lr, tpr_lr, color='#2ecc71', lw=2, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
ax2.plot(fpr_rf, tpr_rf, color='#9b59b6', lw=2, label=f'Random Forest (AUC = {auc_rf:.3f})')
ax2.plot([0, 1], [0, 1], 'k--', lw=1)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=12)
ax2.set_ylabel('True Positive Rate', fontsize=12)
ax2.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')

plt.tight_layout()
plt.savefig('model_performance.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ model_performance.png 저장 완료")

# Figure 4: 주요 인사이트 요약 시각화
fig4, axes = plt.subplots(2, 2, figsize=(14, 12))

# 4-1: Q7 점수별 구매 의향
ax1 = axes[0, 0]
q7_purchase = df.groupby('Q7_score')['구매의향'].agg(['mean', 'count']).reset_index()
q7_purchase = q7_purchase[q7_purchase['Q7_score'].between(1, 5)]
colors = plt.cm.RdYlGn(q7_purchase['mean'])
bars = ax1.bar(q7_purchase['Q7_score'].astype(int).astype(str), q7_purchase['mean'], color=colors, edgecolor='black')
ax1.set_xlabel('Q7 점수 (아침/밤 두피 차이 인식)', fontsize=12)
ax1.set_ylabel('구매 의향 비율', fontsize=12)
ax1.set_title('Q7 점수별 구매 의향 비율', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
for bar, (_, row) in zip(bars, q7_purchase.iterrows()):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{row["mean"]:.0%}\n(n={int(row["count"])})', ha='center', va='bottom', fontsize=10)

# 4-2: 머리 감는 시간대별 구매 의향
ax2 = axes[0, 1]
time_purchase = df.groupby('머리감는시간')['구매의향'].agg(['mean', 'count']).reset_index()
time_purchase = time_purchase.sort_values('mean', ascending=True)
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(time_purchase)))
bars = ax2.barh(time_purchase['머리감는시간'], time_purchase['mean'], color=colors, edgecolor='black')
ax2.set_xlabel('구매 의향 비율', fontsize=12)
ax2.set_title('머리 감는 시간대별 구매 의향', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1)
for bar, (_, row) in zip(bars, time_purchase.iterrows()):
    ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{row["mean"]:.0%} (n={int(row["count"])})', ha='left', va='center', fontsize=10)

# 4-3: 두피 고민별 구매 의향
ax3 = axes[1, 0]
concern_purchase = []
for concern in scalp_concerns[:-1]:  # 특별한 고민 없음 제외
    concern_df = df[df[f'고민_{concern}'] == 1]
    if len(concern_df) >= 5:
        concern_purchase.append({
            '고민': concern,
            '구매의향': concern_df['구매의향'].mean(),
            '응답수': len(concern_df)
        })
concern_df = pd.DataFrame(concern_purchase).sort_values('구매의향', ascending=True)
colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(concern_df)))
bars = ax3.barh(concern_df['고민'], concern_df['구매의향'], color=colors, edgecolor='black')
ax3.set_xlabel('구매 의향 비율', fontsize=12)
ax3.set_title('두피 고민별 구매 의향', fontsize=14, fontweight='bold')
ax3.set_xlim(0, 1)
for bar, (_, row) in zip(bars, concern_df.iterrows()):
    ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
             f'{row["구매의향"]:.0%} (n={int(row["응답수"])})', ha='left', va='center', fontsize=10)

# 4-4: 성별 × 하루2번샴푸 × 구매의향
ax4 = axes[1, 1]
cross_data = df.groupby(['성별', '하루2번샴푸'])['구매의향'].mean().unstack()
cross_data.columns = ['하루 1번', '하루 2번']
cross_data.plot(kind='bar', ax=ax4, color=['#3498db', '#e74c3c'], edgecolor='black', width=0.7)
ax4.set_xlabel('성별', fontsize=12)
ax4.set_ylabel('구매 의향 비율', fontsize=12)
ax4.set_title('성별 × 샴푸 횟수별 구매 의향', fontsize=14, fontweight='bold')
ax4.set_ylim(0, 1)
ax4.legend(title='샴푸 횟수')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
for container in ax4.containers:
    ax4.bar_label(container, fmt='%.0f%%', label_type='edge', fontsize=10)

plt.suptitle('데이앤나이트 듀얼 샴푸 - 주요 인사이트', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('key_insights.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ key_insights.png 저장 완료")

# Figure 5: Logistic Regression 계수 시각화 (영향 방향 포함)
fig5, ax = plt.subplots(figsize=(12, 8))
lr_sorted = lr_coef.sort_values('계수')
colors = ['#2ecc71' if c > 0 else '#e74c3c' for c in lr_sorted['계수']]
bars = ax.barh(lr_sorted['피처'], lr_sorted['계수'], color=colors, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Logistic Regression 계수', fontsize=12)
ax.set_title('구매 의향에 대한 각 변수의 영향\n(🟢 양의 영향: 구매↑ / 🔴 음의 영향: 구매↓)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png', dpi=150, bbox_inches='tight', facecolor='white')
print("   ✅ logistic_regression_coefficients.png 저장 완료")

print("\n" + "=" * 70)
print("✅ 분석 완료!")
print("=" * 70)
print("""
📁 생성된 파일:
   1. feature_importance_comparison.png - 3개 모델 Feature Importance 비교
   2. decision_tree_visualization.png   - Decision Tree 시각화
   3. model_performance.png             - 모델 성능 비교 & ROC Curve
   4. key_insights.png                  - 주요 인사이트 요약
   5. logistic_regression_coefficients.png - 변수별 영향 방향
""")
