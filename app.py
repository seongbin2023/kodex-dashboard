import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# --- 1. 예측 모델 클래스 (데이터 수집 안정화) ---
class KODEX200AdvancedPredictor:
    def __init__(self, start_date='2020-01-01'):
        self.start_date = start_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.data = {}
        self.indicators = {}

    def fetch_data(self, symbol, name, is_krx=False):
        try:
            df = fdr.DataReader(symbol, self.start_date, self.end_date)
            if df is not None and not df.empty: return df
        except: pass
        try:
            yf_sym = symbol if not is_krx else f"{symbol}.KS"
            df = yf.download(yf_sym, start=self.start_date, end=self.end_date, progress=False)
            if not df.empty: return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except: pass
        return None

    def load_all_data(self):
        asset_map = [
            ('069500', 'KODEX200', True),
            ('005930', 'SAMSUNG', True),
            ('^TNX', 'TNX', False),         
            ('USDKRW=X', 'USDKRW', False),  
            ('^VIX', 'VIX', False),         
            ('CL=F', 'WTI_OIL', False),     
            ('GC=F', 'GOLD', False)         
        ]
        for sym, name, is_kr in asset_map:
            res = self.fetch_data(sym, name, is_kr)
            if res is not None: self.data[name] = res

    def calculate_indicators(self):
        for name, df in self.data.items():
            df = df.copy()
            df['Returns_5d'] = df['Close'].pct_change(5)
            df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-10)))

            if name == 'SAMSUNG': df['SAM_Momentum'] = df['Close'].pct_change(5)
            elif name == 'TNX': df['TNX_Momentum'] = df['Close'].pct_change(10)
            elif name == 'WTI_OIL': df['OIL_Momentum'] = df['Close'].pct_change(5)
            elif name == 'GOLD': df['GOLD_Momentum'] = df['Close'].pct_change(5)
            self.indicators[name] = df.fillna(method='bfill').fillna(method='ffill')

    def create_model(self):
        base = self.indicators['KODEX200'][['Close', 'RSI', 'Volatility_20d', 'Returns_5d']].copy()
        for ext in ['TNX', 'VIX', 'USDKRW', 'WTI_OIL', 'GOLD', 'SAMSUNG']:
            if ext in self.indicators:
                ext_df = self.indicators[ext][['Close']].rename(columns={'Close': f'{ext}_val'})
                if ext == 'SAMSUNG': ext_df['SAM_Mom'] = self.indicators['SAMSUNG']['SAM_Momentum']
                elif ext == 'TNX': ext_df['TNX_Mom'] = self.indicators['TNX']['TNX_Momentum']
                elif ext == 'WTI_OIL': ext_df['OIL_Mom'] = self.indicators['WTI_OIL']['OIL_Momentum']
                elif ext == 'GOLD': ext_df['GOLD_Mom'] = self.indicators['GOLD']['GOLD_Momentum']
                base = base.join(ext_df, how='left').ffill()

        base['Target'] = (base['Close'].rolling(12).min().shift(-12) - base['Close']) / base['Close'] * 100
        dataset = base.dropna()
        X = dataset.drop(['Close', 'Target'], axis=1)
        y = dataset['Target']
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
        self.model.fit(self.scaler.fit_transform(X), y)
        self.feature_names = X.columns

# --- 2. UI 레이아웃 최적화 ---
st.set_page_config(page_title="KODEX 200 리스크 진단", page_icon="📈", layout="wide")

st.title("🚨 KODEX 200 통합 리스크 진단 대시보드")
st.markdown("매크로 경제, 삼성전자 모멘텀 및 지정학적 리스크를 통합 분석합니다.")

@st.cache_resource
def get_predictor():
    p = KODEX200AdvancedPredictor()
    p.load_all_data(); p.calculate_indicators(); p.create_model()
    return p

with st.spinner('실시간 시장 데이터를 분석 중입니다...'):
    predictor = get_predictor()

# 리스크 데이터 추출
last_features = []
for feat in predictor.feature_names:
    if '_val' in feat:
        name = feat.replace('_val', '')
        last_features.append(predictor.indicators[name]['Close'].iloc[-1])
    elif 'SAM_Mom' in feat: last_features.append(predictor.indicators['SAMSUNG']['SAM_Momentum'].iloc[-1])
    elif 'TNX_Mom' in feat: last_features.append(predictor.indicators['TNX']['TNX_Momentum'].iloc[-1])
    elif 'OIL_Mom' in feat: last_features.append(predictor.indicators['WTI_OIL']['OIL_Momentum'].iloc[-1])
    elif 'GOLD_Mom' in feat: last_features.append(predictor.indicators['GOLD']['GOLD_Momentum'].iloc[-1])
    else: last_features.append(predictor.indicators['KODEX200'][feat].iloc[-1])

prediction = predictor.model.predict(predictor.scaler.transform([last_features]))[0]

# --- 상단 리포트 구역 ---
st.divider()
c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("🚩 진단 결과")
    level = "🔴 고위험 (HIGH)" if prediction <= -8 else ("🟡 중위험 (MEDIUM)" if prediction <= -4 else "🟢 저위험 (LOW)")
    color = "red" if prediction <= -8 else ("orange" if prediction <= -4 else "green")
    
    st.markdown(f"<h1 style='color: {color}; font-size: 45px;'>{level}</h1>", unsafe_allow_html=True)
    st.metric(label="12일 내 예상 최대 하락폭", value=f"{prediction:.2f}%")

with c2:
    st.subheader("🔍 리스크 주도 요인 분석")
    term_map = {
        'SAM_Mom': '삼성전자 모멘텀', 'SAM_val': '삼성전자 주가',
        'TNX_Mom': '금리 변화 속도', 'TNX_val': '미 국채 금리',
        'OIL_Mom': '유가 급등폭', 'WTI_OIL_val': '국제 유가',
        'GOLD_Mom': '금값 급등폭', 'GOLD_val': '금 시세',
        'USDKRW_val': '원/달러 환율', 'VIX_val': '시장 공포(VIX)',
        'RSI': 'KODEX200 RSI', 'Volatility_20d': '지수 변동성', 'Returns_5d': '지수 단기수익률'
    }
    
    importances = pd.Series(predictor.model.feature_importances_, index=predictor.feature_names).sort_values(ascending=False).head(6)
    importances.index = [term_map.get(x, x) for x in importances.index]
    
    # 가로형 바 차트로 변경하여 가독성 향상
    st.bar_chart(importances * 100)

st.divider()
st.info("💡 **알림:** 삼성전자 모멘텀은 지수 하락의 선행 지표 역할을 합니다. 금리 변화 속도가 높을 경우 기술주와 지수 전체에 강한 하방 압력이 작용합니다.")
