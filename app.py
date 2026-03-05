import streamlit as st
import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')

# --- 1. 예측 모델 클래스 (지정학적 리스크 포함) ---
class KODEX200AdvancedPredictor:
    def __init__(self, start_date='2020-01-01'):
        self.start_date = start_date
        self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.data = {}
        self.indicators = {}
        self.crash_periods = []

    def fetch_data(self, symbol, name, is_krx=False):
        try:
            df = fdr.DataReader(symbol, self.start_date, self.end_date)
            if df is not None and not df.empty:
                return df
        except: pass

        try:
            yf_sym = symbol if not is_krx else f"{symbol}.KS"
            df = yf.download(yf_sym, start=self.start_date, end=self.end_date, progress=False)
            if not df.empty:
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                return df
        except: pass
        return None

    def load_all_data(self):
        asset_map = [
            ('069500', 'KODEX200', True),
            ('^TNX', 'TNX', False),         
            ('USDKRW=X', 'USDKRW', False),  
            ('^VIX', 'VIX', False),         
            ('CL=F', 'WTI_OIL', False),     
            ('GC=F', 'GOLD', False)         
        ]
        for sym, name, is_kr in asset_map:
            result = self.fetch_data(sym, name, is_kr)
            if result is not None:
                self.data[name] = result

    def calculate_indicators(self):
        for name, df in self.data.items():
            df = df.copy()
            df['MA_20'] = df['Close'].rolling(20).mean()
            df['MA_60'] = df['Close'].rolling(60).mean()
            
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['RSI'] = 100 - (100 / (1 + gain/(loss + 1e-10)))
            
            df['Returns_5d'] = df['Close'].pct_change(5)
            df['Volatility_20d'] = df['Close'].pct_change().rolling(20).std() * np.sqrt(252)
            
            if name == 'TNX':
                df['TNX_Momentum'] = df['Close'].pct_change(10)
            elif name == 'WTI_OIL':
                df['OIL_Momentum'] = df['Close'].pct_change(5)
            elif name == 'GOLD':
                df['GOLD_Momentum'] = df['Close'].pct_change(5)
                
            self.indicators[name] = df.fillna(method='bfill')

    def create_model(self):
        base = self.indicators['KODEX200'][['Close', 'RSI', 'Volatility_20d', 'Returns_5d']].copy()
        for ext in ['TNX', 'VIX', 'USDKRW', 'WTI_OIL', 'GOLD']:
            if ext in self.indicators:
                ext_df = self.indicators[ext][['Close']].rename(columns={'Close': f'{ext}_val'})
                if ext == 'TNX':
                    ext_df['TNX_Mom'] = self.indicators['TNX']['TNX_Momentum']
                elif ext == 'WTI_OIL':
                    ext_df['OIL_Mom'] = self.indicators['WTI_OIL']['OIL_Momentum']
                elif ext == 'GOLD':
                    ext_df['GOLD_Mom'] = self.indicators['GOLD']['GOLD_Momentum']
                base = base.join(ext_df, how='left').ffill()

        base['Target'] = (base['Close'].rolling(12).min().shift(-12) - base['Close']) / base['Close'] * 100
        dataset = base.dropna()

        X = dataset.drop(['Close', 'Target'], axis=1)
        y = dataset['Target']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns

        self.model = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=42)
        self.model.fit(X_scaled, y)


# --- 2. Streamlit 웹 앱 UI 구성 ---
st.set_page_config(page_title="KODEX 200 리스크 진단", page_icon="📈", layout="wide")

st.title("🚨 KODEX 200 통합 리스크 진단 대시보드")
st.markdown("매크로 경제 및 지정학적 리스크(유가, 금, 환율)를 통합 분석하여 하락 위험을 예측합니다.")

# 데이터 로딩 및 분석 (캐싱을 통해 앱 새로고침 시 속도 대폭 향상)
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

# 리스크 진단 결과 추출
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
    level, color, icon = "고위험 (HIGH RISK)", "red", "🔴"
    action_plan = "현재 시장은 금리 및 지정학적 리스크로 인해 매우 불안정한 상태입니다. 주식 비중을 즉시 축소하고 현금 및 안전자산(달러, 금) 비중 확대를 적극 권장합니다."
elif prediction <= -4:
    level, color, icon = "중위험 (MEDIUM RISK)", "orange", "🟡"
    action_plan = "매크로 변동성이 커질 수 있는 경계 구간입니다. 유가와 환율 추이를 예의주시하며, 보유 종목의 손절가를 타이트하게 설정하십시오."
else:
    level, color, icon = "저위험 (LOW RISK)", "green", "🟢"
    action_plan = "현재 지정학적 돌발 변수나 매크로 하락 위험은 통제 가능한 수준입니다. 기존의 투자 전략을 유지하며 우량주 중심의 투자가 유효합니다."

# 화면 출력 (UI 구성)
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

# 리스크 주도 요인 차트화
st.subheader("🔍 현재 하락 리스크를 주도하는 핵심 요인 Top 5")

term_map = {
    'RSI': 'KODEX200 RSI',
    'Volatility_20d': 'KODEX200 변동성',
    'Returns_5d': 'KODEX200 단기 수익률',
    'TNX_val': '국채 10년물 금리',
    'TNX_Mom': '금리 변화 속도 (긴축 리스크)',
    'VIX_val': 'VIX (공포 지수)',
    'USDKRW_val': '원/달러 환율',
    'WTI_OIL_val': '국제 유가',
    'OIL_Mom': '유가 급등폭 (지정학 리스크)',
    'GOLD_val': '국제 금 시세',
    'GOLD_Mom': '금값 급등폭 (안전자산 쏠림)'
}

importances = pd.Series(predictor.model.feature_importances_, index=predictor.feature_names).sort_values(ascending=False).head(5)
importances.index = [term_map.get(x, x) for x in importances.index]

st.bar_chart(importances * 100)