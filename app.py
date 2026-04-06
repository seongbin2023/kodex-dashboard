import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# 기존에 만드신 예측 클래스를 가져옵니다. 
# (만약 한 파일에 합치려면 여기에 KODEX200AdvancedPredictor 클래스 코드를 전체 복사해 넣으세요)
from predictor_model import KODEX200AdvancedPredictor 

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="KODEX 200 리스크 진단 앱", page_icon="📈", layout="wide")

st.title("🚨 KODEX 200 통합 리스크 진단 대시보드")
st.markdown("매크로 경제 및 지정학적 리스크(유가, 금, 환율)를 통합 분석하여 하락 위험을 예측합니다.")

# --- 데이터 로딩 및 분석 (캐싱을 통해 앱 속도 향상) ---
@st.cache_resource
def get_predictor():
    p = KODEX200AdvancedPredictor()
    p.load_all_data()
    p.calculate_indicators()
    p.create_model()
    return p

with st.spinner('글로벌 금융 데이터 및 지정학적 리스크 지표를 수집하고 분석 중입니다...'):
    predictor = get_predictor()

st.success('데이터 로딩 및 AI 모델 학습이 완료되었습니다!')

st.divider()

# --- 리스크 진단 결과 추출 로직 ---
last_features = []
for feat in predictor.feature_names:
    if '_val' in feat:
        name = feat.replace('_val', '')
        last_features.append(predictor.indicators[name]['Close'].iloc[-1])
    elif 'TNX_Mom' in feat:
        last_features.append(predictor.indicators['TNX']['TNX_Momentum'].iloc[-1])
    elif 'OIL_Mom' in feat:
        last_features.append(predictor.indicators['WTI_OIL']['OIL_Momentum'].iloc[-1])
    elif 'GOLD_Mom' in feat:
        last_features.append(predictor.indicators['GOLD']['GOLD_Momentum'].iloc[-1])
    else:
        last_features.append(predictor.indicators['KODEX200'][feat].iloc[-1])

scaled_feat = predictor.scaler.transform([last_features])
prediction = predictor.model.predict(scaled_feat)[0]

# 위험 등급 분류
if prediction <= -8:
    level, color, icon = "🔴 고위험 (HIGH RISK)", "red", "🚨"
    action_plan = "현재 시장은 금리 및 지정학적 리스크로 인해 매우 불안정한 상태입니다. 주식 비중을 즉시 축소하고 현금 및 안전자산(달러, 금) 비중 확대를 적극 권장합니다."
elif prediction <= -4:
    level, color, icon = "🟡 중위험 (MEDIUM RISK)", "orange", "⚠️"
    action_plan = "매크로 변동성이 커질 수 있는 경계 구간입니다. 유가와 환율 추이를 예의주시하며, 보유 종목의 손절가를 타이트하게 설정하십시오."
else:
    level, color, icon = "🟢 저위험 (LOW RISK)", "green", "✅"
    action_plan = "현재 지정학적 돌발 변수나 매크로 하락 위험은 통제 가능한 수준입니다. 기존의 투자 전략을 유지하며 우량주 중심의 투자가 유효합니다."

# --- 화면 출력 (UI 구성) ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"{icon} 현재 시장 위험 등급")
    st.markdown(f"<h2 style='color: {color};'>{level}</h2>", unsafe_allow_html=True)
    
    st.subheader("📉 12일 내 예상 최대 하락폭")
    st.metric(label="Expected Maximum Drawdown", value=f"{prediction:.2f}%")

with col2:
    st.subheader("💡 투자 행동 지침")
    st.info(action_plan)

st.divider()

# --- 리스크 주도 요인 차트화 ---
st.subheader("🔍 현재 하락 리스크를 주도하는 핵심 요인 Top 5")

term_map = {
    'RSI': 'KODEX200 RSI',
    'Volatility_20d': 'KODEX200 변동성',
    'Returns_5d': 'KODEX200 단기 수익률',
    'TNX_val': '국채 10년물 금리',
    'TNX_Mom': '금리 변화 속도',
    'VIX_val': 'VIX (공포 지수)',
    'USDKRW_val': '원/달러 환율',
    'WTI_OIL_val': '국제 유가',
    'OIL_Mom': '유가 급등폭 (지정학 리스크)',
    'GOLD_val': '국제 금 시세',
    'GOLD_Mom': '금값 급등폭 (안전자산 쏠림)'
}

importances = pd.Series(predictor.model.feature_importances_, index=predictor.feature_names).sort_values(ascending=False).head(5)
importances.index = [term_map.get(x, x) for x in importances.index]

# 스트림릿 내장 바 차트 활용
st.bar_chart(importances * 100)
