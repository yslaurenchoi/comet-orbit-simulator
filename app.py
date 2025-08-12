import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# 페이지 설정
st.set_page_config(
    page_title="혜성 궤도 시뮬레이터",
    page_icon="🌟",
    layout="wide"
)

# 천문학적 상수
AU = 1.496e11  # 천문단위 (미터)
G = 6.67430e-11  # 중력상수 (m³/kg/s²)
M_sun = 1.989e30  # 태양질량 (kg)
YEAR = 365.25 * 24 * 3600  # 1년 (초)

class CometOrbitSimulator:
    def __init__(self, star_mass, comet_mass, eccentricity, 
                 semi_major_axis, mass_loss_rate):
        """
        혜성 궤도 시뮬레이터 초기화
        
        Parameters:
        - star_mass: 항성 질량 (태양질량 단위)
        - comet_mass: 혜성 초기 질량 (kg)
        - eccentricity: 이심률 (0-2) - 고정값
        - semi_major_axis: 긴반지름 (AU) - 고정값
        - mass_loss_rate: 질량 소실률 (kg/s)
        """
        self.star_mass = star_mass * M_sun
        self.initial_comet_mass = comet_mass
        self.current_comet_mass = comet_mass
        self.eccentricity = eccentricity  # 고정된 이심률
        self.semi_major_axis = semi_major_axis * AU  # 고정된 긴반지름
        self.mass_loss_rate = mass_loss_rate
        self.is_extinct = False  # 혜성 소멸 여부
        self.extinction_time = None  # 소멸 시간
        
        # 궤도 주기 계산 (케플러 제3법칙 - 타원 궤도만)
        if self.eccentricity < 1:
            self.orbital_period = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / (G * self.star_mass))
        else:
            self.orbital_period = np.inf  # 포물선/쌍곡선 궤도는 주기가 없음
        
    def calculate_orbital_velocity(self, r):
        """주어진 거리에서의 궤도 속도 계산"""
        return np.sqrt(G * self.star_mass * (2/r - 1/self.semi_major_axis))
    
    def update_mass(self, time_step, current_time):
        """질량 소실 계산 (궤도에는 영향 없음)"""
        if self.is_extinct:
            return 0  # 이미 소멸된 경우
        
        # 질량 소실량 계산
        mass_loss = self.mass_loss_rate * time_step
        
        # 질량이 0 이하로 떨어지는지 확인
        if self.current_comet_mass - mass_loss <= 0:
            self.current_comet_mass = 0
            self.is_extinct = True
            self.extinction_time = current_time
            return 0
        
        self.current_comet_mass -= mass_loss
        
        # 현재 질량 비율 반환 (시각화용)
        return self.current_comet_mass / self.initial_comet_mass
    
    def get_orbital_position(self, time):
        """주어진 시간에서의 궤도 위치 계산 (모든 이심률 지원)"""
        if self.is_extinct:
            return None, None, None  # 소멸된 혜성은 위치가 없음
        
        # 이심률에 따른 궤도 계산 분기
        if self.eccentricity < 1:
            # 타원 궤도 (e < 1)
            mean_anomaly = 2 * np.pi * time / self.orbital_period
            eccentric_anomaly = self.solve_kepler_equation(mean_anomaly, self.eccentricity)
            
            # 참 근점 이상 (True Anomaly)
            true_anomaly = 2 * np.arctan(np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)) * 
                                         np.tan(eccentric_anomaly / 2))
            
            # 궤도 반지름
            r = self.semi_major_axis * (1 - self.eccentricity**2) / (1 + self.eccentricity * np.cos(true_anomaly))
            
        elif self.eccentricity == 1:
            # 포물선 궤도 (e = 1)
            # 포물선 궤도에서는 시간-위치 관계가 다름
            n = np.sqrt(G * self.star_mass / (2 * self.semi_major_axis**3))
            D = n * time  # Mean anomaly for parabolic orbit
            
            # 포물선 이상 (Parabolic Anomaly) 계산
            E = self.solve_parabolic_equation(D)
            true_anomaly = 2 * np.arctan(E)
            
            # 궤도 반지름 (포물선)
            r = self.semi_major_axis * (1 + E**2)
            
        else:
            # 쌍곡선 궤도 (e > 1)
            n = np.sqrt(G * self.star_mass / (-self.semi_major_axis**3))  # 음수 값
            mean_anomaly = n * time
            
            # 쌍곡선 이상 (Hyperbolic Anomaly) 계산
            hyperbolic_anomaly = self.solve_hyperbolic_equation(mean_anomaly, self.eccentricity)
            
            # 참 근점 이상
            true_anomaly = 2 * np.arctan(np.sqrt((self.eccentricity + 1) / (self.eccentricity - 1)) * 
                                         np.tanh(hyperbolic_anomaly / 2))
            
            # 궤도 반지름 (쌍곡선)
            r = self.semi_major_axis * (self.eccentricity**2 - 1) / (1 + self.eccentricity * np.cos(true_anomaly))
        
        # 직교 좌표계로 변환
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        return x, y, r
    
    def solve_kepler_equation(self, mean_anomaly, eccentricity, tolerance=1e-10):
        """케플러 방정식을 뉴턴 방법으로 해결 (타원 궤도용)"""
        eccentric_anomaly = mean_anomaly
        
        for _ in range(100):
            f = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
            if abs(f) < tolerance:
                break
            df = 1 - eccentricity * np.cos(eccentric_anomaly)
            eccentric_anomaly = eccentric_anomaly - f / df
        
        return eccentric_anomaly
    
    def solve_parabolic_equation(self, D, tolerance=1e-10):
        """포물선 궤도 방정식 해결"""
        # Barker's equation: D = E + E^3/3
        E = D  # 초기 추정값
        
        for _ in range(100):
            f = E + E**3/3 - D
            if abs(f) < tolerance:
                break
            df = 1 + E**2
            E = E - f / df
        
        return E
    
    def solve_hyperbolic_equation(self, mean_anomaly, eccentricity, tolerance=1e-10):
        """쌍곡선 궤도 방정식 해결"""
        # 쌍곡선 케플러 방정식: M = e*sinh(H) - H
        H = mean_anomaly  # 초기 추정값
        
        for _ in range(100):
            f = eccentricity * np.sinh(H) - H - mean_anomaly
            if abs(f) < tolerance:
                break
            df = eccentricity * np.cosh(H) - 1
            if abs(df) < tolerance:
                break
            H = H - f / df
        
        return H
    
    def generate_orbit_data(self, total_time, time_steps):
        """전체 궤도 데이터 생성"""
        times = np.linspace(0, total_time, time_steps)
        positions = []
        masses = []
        
        for i, t in enumerate(times):
            if i > 0:
                # 질량 소실 계산 (궤도에는 영향 없음)
                time_step = times[i] - times[i-1]
                mass_ratio = self.update_mass(time_step, t)
            
            # 현재 위치 계산 (고정된 궤도)
            x, y, r = self.get_orbital_position(t)
            
            if self.is_extinct and x is None:
                # 혜성이 소멸된 경우 시뮬레이션 종료
                break
            
            positions.append((x, y))
            masses.append(self.current_comet_mass)
        
        # 실제 시뮬레이션된 시간만 반환
        actual_times = times[:len(positions)]
        
        return actual_times, positions, masses

def main():
    # 타이틀과 설명
    st.title("🌟 혜성 궤도 시뮬레이터")
    st.markdown("혜성의 질량 소실 과정을 고정된 궤도에서 시뮬레이션합니다.")
    
    # 사이드바 - 입력 매개변수
    st.sidebar.header("🔧 시뮬레이션 매개변수")
    
    # 기본값 설정 섹션
    st.sidebar.markdown("### 기본 설정")
    
    # 항성 질량 (태양질량 단위)
    star_mass = st.sidebar.slider(
        "항성 질량 (태양질량 단위)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="항성의 질량을 태양질량 단위로 입력하세요. 태양 = 1.0"
    )
    
    # 혜성 초기 질량 (kg)
    comet_mass_exp = st.sidebar.slider(
        "혜성 초기 질량 (10^x kg)",
        min_value=10,
        max_value=15,
        value=12,
        step=1,
        help="혜성의 초기 질량을 10의 거듭제곱으로 설정하세요."
    )
    comet_mass = 10**comet_mass_exp
    
    # 궤도 이심률 (확장된 범위 - 포물선, 쌍곡선 포함)
    eccentricity = st.sidebar.slider(
        "궤도 이심률",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="0: 원궤도, 0<e<1: 타원궤도, e=1: 포물선궤도, e>1: 쌍곡선궤도"
    )
    
    # 이심률에 따른 궤도 타입 표시
    if eccentricity == 0:
        orbit_type = "원궤도 (Circle)"
    elif 0 < eccentricity < 1:
        orbit_type = "타원궤도 (Ellipse)"
    elif eccentricity == 1:
        orbit_type = "포물선궤도 (Parabola)"
    else:  # eccentricity > 1
        orbit_type = "쌍곡선궤도 (Hyperbola)"
    
    st.sidebar.markdown(f"**궤도 타입:** {orbit_type}")
    
    # 긴반지름 (AU, 고정값)
    semi_major_axis = st.sidebar.slider(
        "긴반지름 (AU)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="궤도의 긴반지름을 천문단위(AU)로 입력하세요. (시뮬레이션 중 변하지 않음)"
    )
    
    # 질량 소실률 (kg/s) - 더 작은 범위
    mass_loss_exp = st.sidebar.slider(
        "질량 소실률 (10^x kg/s)",
        min_value=1,
        max_value=5,
        value=3,
        step=0.5,
        help="혜성이 초당 잃는 질량을 10의 거듭제곱으로 설정하세요."
    )
    mass_loss_rate = 10**mass_loss_exp
    
    # 쌍곡선/포물선 궤도 경고
    if eccentricity >= 1:
        st.sidebar.warning("⚠️ e≥1: 비주기 궤도 (무한대로 날아감)")
        # 쌍곡선/포물선 궤도의 경우 시뮬레이션 시간을 제한
        max_sim_years = 5
        sim_years_default = 1
    else:
        max_sim_years = 200
        estimated_lifetime = comet_mass / mass_loss_rate / YEAR
        sim_years_default = min(int(estimated_lifetime * 1.5), 50)  # 생존시간의 1.5배 또는 최대 50년
    
    # 시뮬레이션 시간 설정 (자동 설정)
    st.sidebar.markdown("### 시뮬레이션 설정")
    sim_years = st.sidebar.slider(
        "시뮬레이션 기간 (년)",
        min_value=1,
        max_value=max_sim_years,
        value=sim_years_default,
        help="시뮬레이션할 기간을 년 단위로 설정하세요. (예상 생존시간에 맞춰 자동 조정됨)"
    )
    
    # 혜성 생존 시간 예측
    estimated_lifetime = comet_mass / mass_loss_rate / YEAR
    st.sidebar.markdown(f"### 🔮 예상 혜성 생존시간: {estimated_lifetime:.1f}년")
    
    if eccentricity < 1:
        if estimated_lifetime < sim_years:
            st.sidebar.warning(f"⚠️ 혜성이 {estimated_lifetime:.1f}년 후 완전히 소멸됩니다!")
    else:
        st.sidebar.info("📌 비주기 궤도: 혜성이 무한대로 멀어집니다")
    
    # 현재 설정 표시
    st.sidebar.markdown("### 📊 현재 설정값")
    st.sidebar.write(f"**항성 질량:** {star_mass:.1f} 태양질량")
    st.sidebar.write(f"**혜성 질량:** {comet_mass:.1e} kg")
    st.sidebar.write(f"**이심률:** {eccentricity:.2f} ({orbit_type})")
    st.sidebar.write(f"**긴반지름:** {semi_major_axis:.1f} AU")
    st.sidebar.write(f"**질량소실률:** {mass_loss_rate:.1e} kg/s")
    st.sidebar.write(f"**시뮬레이션 기간:** {sim_years} 년")
    
    # 시뮬레이션 실행
    if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
        # 시뮬레이터 초기화
        simulator = CometOrbitSimulator(
            star_mass=star_mass,
            comet_mass=comet_mass,
            eccentricity=eccentricity,
            semi_major_axis=semi_major_axis,
            mass_loss_rate=mass_loss_rate
        )
        
        # 시뮬레이션 데이터 생성
        total_time = sim_years * YEAR
        time_steps = 1000
        
        with st.spinner("시뮬레이션 계산 중..."):
            times, positions, masses = simulator.generate_orbit_data(total_time, time_steps)
        
        # 혜성 소멸 여부 확인
        if simulator.is_extinct:
            st.warning(f"🔥 **혜성이 {simulator.extinction_time/YEAR:.1f}년 후 완전히 소멸되었습니다!**")
        
        # 결과 표시
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌍 궤도 애니메이션")
            
            # 궤도 애니메이션 생성
            fig = go.Figure()
            
            # 항성 추가
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=25, color='gold', symbol='star'),
                name='항성',
                hovertemplate='<b>항성</b><br>질량: %.1f 태양질량<extra></extra>' % star_mass
            ))
            
            # 궤도 경로 추가 (완전한 타원 궤도)
            x_pos = [pos[0]/AU for pos in positions]
            y_pos = [pos[1]/AU for pos in positions]
            
            fig.add_trace(go.Scatter(
                x=x_pos, y=y_pos,
                mode='lines',
                line=dict(color='lightblue', width=2),
                name='궤도 경로',
                hovertemplate='궤도 경로<extra></extra>'
            ))
            
            # 혜성 위치 (애니메이션)
            frames = []
            for i in range(0, len(positions), max(1, len(positions)//100)):  # 100 프레임으로 제한
                frame_data = []
                
                # 항성
                frame_data.append(go.Scatter(
                    x=[0], y=[0],
                    mode='markers',
                    marker=dict(size=25, color='gold', symbol='star'),
                    name='항성'
                ))
                
                # 궤도 경로 (현재까지)
                frame_data.append(go.Scatter(
                    x=x_pos[:i+1], y=y_pos[:i+1],
                    mode='lines',
                    line=dict(color='lightblue', width=2),
                    name='궤도 경로'
                ))
                
                # 혜성 현재 위치 (질량이 0이 아닐 때만 표시)
                if masses[i] > 0:
                    comet_size = max(8, 20 * masses[i] / comet_mass)  # 질량에 따른 크기 변화
                    comet_color = 'red' if masses[i] > comet_mass * 0.1 else 'orange'  # 질량에 따른 색상 변화
                    
                    frame_data.append(go.Scatter(
                        x=[x_pos[i]], y=[y_pos[i]],
                        mode='markers',
                        marker=dict(size=comet_size, color=comet_color, symbol='circle'),
                        name='혜성',
                        hovertemplate=f'<b>혜성</b><br>시간: {times[i]/YEAR:.1f}년<br>질량: {masses[i]:.2e} kg<br>이심률: {eccentricity:.3f} ({orbit_type})<extra></extra>'
                    ))
                else:
                    # 혜성이 소멸된 경우 소멸 위치에 X 표시
                    frame_data.append(go.Scatter(
                        x=[x_pos[i]], y=[y_pos[i]],
                        mode='markers',
                        marker=dict(size=15, color='gray', symbol='x'),
                        name='소멸된 혜성',
                        hovertemplate=f'<b>혜성 소멸</b><br>시간: {times[i]/YEAR:.1f}년<br>질량: 0 kg<extra></extra>'
                    ))
                
                frames.append(go.Frame(data=frame_data, name=str(i)))
            
            fig.frames = frames
            
            # 레이아웃 설정
            fig.update_layout(
                title=f"혜성 궤도 시뮬레이션 ({orbit_type})",
                xaxis_title="거리 (AU)",
                yaxis_title="거리 (AU)",
                showlegend=True,
                width=700,
                height=600,
                xaxis=dict(scaleanchor="y", scaleratio=1),
                plot_bgcolor='rgba(0,0,0,0.05)',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'x': 0.1,
                    'y': 0,
                    'xanchor': 'right',
                    'yanchor': 'top',
                    'buttons': [
                        {
                            'label': '▶️ 재생',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 100, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 50}
                            }]
                        },
                        {
                            'label': '⏸️ 일시정지',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }}}])
