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
    def __init__(self, star_mass, comet_mass, initial_eccentricity, 
                 semi_major_axis, mass_loss_rate):
        """
        혜성 궤도 시뮬레이터 초기화
        
        Parameters:
        - star_mass: 항성 질량 (태양질량 단위)
        - comet_mass: 혜성 초기 질량 (kg)
        - initial_eccentricity: 초기 이심률 (0-1)
        - semi_major_axis: 긴반지름 (AU)
        - mass_loss_rate: 질량 소실률 (kg/s)
        """
        self.star_mass = star_mass * M_sun
        self.initial_comet_mass = comet_mass
        self.current_comet_mass = comet_mass
        self.initial_eccentricity = initial_eccentricity
        self.current_eccentricity = initial_eccentricity
        self.semi_major_axis = semi_major_axis * AU
        self.mass_loss_rate = mass_loss_rate
        
        # 궤도 주기 계산 (케플러 제3법칙)
        self.orbital_period = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / (G * self.star_mass))
        
    def calculate_orbital_velocity(self, r):
        """주어진 거리에서의 궤도 속도 계산"""
        return np.sqrt(G * self.star_mass * (2/r - 1/self.semi_major_axis))
    
    def calculate_mass_loss_effect(self, time_step):
        """질량 소실이 궤도에 미치는 영향 계산"""
        # 질량 소실량 계산
        mass_loss = self.mass_loss_rate * time_step
        
        # 질량이 0 이하로 떨어지지 않도록 제한
        if self.current_comet_mass - mass_loss <= 0:
            mass_loss = self.current_comet_mass * 0.99
        
        self.current_comet_mass -= mass_loss
        
        # 질량 소실에 따른 궤도 변화
        # 질량이 줄어들면 태양풍 압력에 더 민감해져 궤도가 변함
        mass_ratio = self.current_comet_mass / self.initial_comet_mass
        
        # 이심률 증가 (질량 소실로 인한 궤도 불안정성)
        eccentricity_increase = (1 - mass_ratio) * 0.1
        self.current_eccentricity = min(0.99, self.initial_eccentricity + eccentricity_increase)
        
        # 긴반지름 변화 (미세한 궤도 확장)
        self.semi_major_axis *= (1 + (1 - mass_ratio) * 0.001)
        
        return mass_ratio
    
    def get_orbital_position(self, time):
        """주어진 시간에서의 궤도 위치 계산"""
        # 평균 근점 이상 (Mean Anomaly)
        mean_anomaly = 2 * np.pi * time / self.orbital_period
        
        # 이심 근점 이상 (Eccentric Anomaly) - 뉴턴 방법으로 해결
        eccentric_anomaly = self.solve_kepler_equation(mean_anomaly, self.current_eccentricity)
        
        # 참 근점 이상 (True Anomaly)
        true_anomaly = 2 * np.arctan(np.sqrt((1 + self.current_eccentricity) / (1 - self.current_eccentricity)) * 
                                     np.tan(eccentric_anomaly / 2))
        
        # 궤도 반지름
        r = self.semi_major_axis * (1 - self.current_eccentricity**2) / (1 + self.current_eccentricity * np.cos(true_anomaly))
        
        # 직교 좌표계로 변환
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        return x, y, r
    
    def solve_kepler_equation(self, mean_anomaly, eccentricity, tolerance=1e-10):
        """케플러 방정식을 뉴턴 방법으로 해결"""
        eccentric_anomaly = mean_anomaly
        
        for _ in range(100):
            f = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
            if abs(f) < tolerance:
                break
            df = 1 - eccentricity * np.cos(eccentric_anomaly)
            eccentric_anomaly = eccentric_anomaly - f / df
        
        return eccentric_anomaly
    
    def generate_orbit_data(self, total_time, time_steps):
        """전체 궤도 데이터 생성"""
        times = np.linspace(0, total_time, time_steps)
        positions = []
        eccentricities = []
        masses = []
        
        for i, t in enumerate(times):
            if i > 0:
                # 질량 소실 효과 적용
                time_step = times[i] - times[i-1]
                mass_ratio = self.calculate_mass_loss_effect(time_step)
            
            # 현재 위치 계산
            x, y, r = self.get_orbital_position(t)
            positions.append((x, y))
            eccentricities.append(self.current_eccentricity)
            masses.append(self.current_comet_mass)
        
        return times, positions, eccentricities, masses

def main():
    # 타이틀과 설명
    st.title("🌟 혜성 궤도 시뮬레이터")
    st.markdown("혜성의 질량 소실이 궤도에 미치는 영향을 시뮬레이션합니다.")
    
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
    
    # 초기 이심률
    initial_eccentricity = st.sidebar.slider(
        "초기 궤도 이심률",
        min_value=0.0,
        max_value=0.99,
        value=0.5,
        step=0.01,
        help="0: 완전한 원궤도, 1에 가까울수록 매우 긴 타원궤도"
    )
    
    # 긴반지름 (AU)
    semi_major_axis = st.sidebar.slider(
        "긴반지름 (AU)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="궤도의 긴반지름을 천문단위(AU)로 입력하세요. 지구-태양 거리 = 1AU"
    )
    
    # 질량 소실률 (kg/s)
    mass_loss_exp = st.sidebar.slider(
        "질량 소실률 (10^x kg/s)",
        min_value=3,
        max_value=8,
        value=6,
        step=1,
        help="혜성이 초당 잃는 질량을 10의 거듭제곱으로 설정하세요."
    )
    mass_loss_rate = 10**mass_loss_exp
    
    # 시뮬레이션 시간 설정
    st.sidebar.markdown("### 시뮬레이션 설정")
    sim_years = st.sidebar.slider(
        "시뮬레이션 기간 (년)",
        min_value=1,
        max_value=100,
        value=10,
        help="시뮬레이션할 기간을 년 단위로 설정하세요."
    )
    
    # 현재 설정 표시
    st.sidebar.markdown("### 📊 현재 설정값")
    st.sidebar.write(f"**항성 질량:** {star_mass:.1f} 태양질량")
    st.sidebar.write(f"**혜성 질량:** {comet_mass:.1e} kg")
    st.sidebar.write(f"**이심률:** {initial_eccentricity:.2f}")
    st.sidebar.write(f"**긴반지름:** {semi_major_axis:.1f} AU")
    st.sidebar.write(f"**질량소실률:** {mass_loss_rate:.1e} kg/s")
    st.sidebar.write(f"**시뮬레이션 기간:** {sim_years} 년")
    
    # 시뮬레이션 실행
    if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
        # 시뮬레이터 초기화
        simulator = CometOrbitSimulator(
            star_mass=star_mass,
            comet_mass=comet_mass,
            initial_eccentricity=initial_eccentricity,
            semi_major_axis=semi_major_axis,
            mass_loss_rate=mass_loss_rate
        )
        
        # 시뮬레이션 데이터 생성
        total_time = sim_years * YEAR
        time_steps = 1000
        
        with st.spinner("시뮬레이션 계산 중..."):
            times, positions, eccentricities, masses = simulator.generate_orbit_data(total_time, time_steps)
        
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
            
            # 궤도 경로 추가
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
                
                # 혜성 현재 위치
                comet_size = max(8, 20 * masses[i] / comet_mass)  # 질량에 따른 크기 변화
                frame_data.append(go.Scatter(
                    x=[x_pos[i]], y=[y_pos[i]],
                    mode='markers',
                    marker=dict(size=comet_size, color='red', symbol='circle'),
                    name='혜성',
                    hovertemplate=f'<b>혜성</b><br>시간: {times[i]/YEAR:.1f}년<br>질량: {masses[i]:.2e} kg<br>이심률: {eccentricities[i]:.3f}<extra></extra>'
                ))
                
                frames.append(go.Frame(data=frame_data, name=str(i)))
            
            fig.frames = frames
            
            # 레이아웃 설정
            fig.update_layout(
                title="혜성 궤도 시뮬레이션",
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
                            }]
                        }
                    ]
                }]
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 궤도 변화 분석")
            
            # 이심률 변화 그래프
            fig_ecc = go.Figure()
            fig_ecc.add_trace(go.Scatter(
                x=[t/YEAR for t in times],
                y=eccentricities,
                mode='lines',
                name='이심률',
                line=dict(color='green', width=3)
            ))
            fig_ecc.update_layout(
                title="이심률 변화",
                xaxis_title="시간 (년)",
                yaxis_title="이심률",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_ecc, use_container_width=True)
            
            # 질량 변화 그래프
            fig_mass = go.Figure()
            fig_mass.add_trace(go.Scatter(
                x=[t/YEAR for t in times],
                y=masses,
                mode='lines',
                name='질량',
                line=dict(color='red', width=3)
            ))
            fig_mass.update_layout(
                title="혜성 질량 변화",
                xaxis_title="시간 (년)",
                yaxis_title="질량 (kg)",
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig_mass, use_container_width=True)
        
        # 시뮬레이션 결과 요약
        st.subheader("📊 시뮬레이션 결과 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "최종 이심률",
                f"{eccentricities[-1]:.3f}",
                f"{eccentricities[-1] - initial_eccentricity:.3f}"
            )
        
        with col2:
            st.metric(
                "최종 질량",
                f"{masses[-1]:.2e} kg",
                f"{masses[-1] - comet_mass:.2e} kg"
            )
        
        with col3:
            mass_loss_percent = (comet_mass - masses[-1]) / comet_mass * 100
            st.metric(
                "질량 소실률",
                f"{mass_loss_percent:.1f}%"
            )
        
        with col4:
            orbital_period_final = 2 * np.pi * np.sqrt(simulator.semi_major_axis**3 / (G * simulator.star_mass))
            st.metric(
                "궤도 주기",
                f"{orbital_period_final / YEAR:.1f} 년"
            )
        
        # 물리학적 해석
        st.subheader("🔬 물리학적 해석")
        
        interpretation = f"""
        **질량 소실 효과:**
        - 혜성이 {mass_loss_percent:.1f}%의 질량을 잃었습니다.
        - 이로 인해 궤도 이심률이 {eccentricities[-1] - initial_eccentricity:.3f} 증가했습니다.
        - 질량 소실은 태양풍 압력에 대한 민감도를 증가시켜 궤도를 불안정하게 만듭니다.
        
        **궤도 특성:**
        - 현재 궤도 주기: {orbital_period_final / YEAR:.1f} 년
        - 최종 이심률: {eccentricities[-1]:.3f} (0=원궤도, 1=포물선궤도)
        - 질량 소실률: {mass_loss_rate:.1e} kg/s
        """
        
        st.markdown(interpretation)
    
    # 도움말 섹션
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 도움말")
    st.sidebar.markdown("""
    **매개변수 설명:**
    - **항성 질량**: 중심별의 질량 (태양 = 1.0)
    - **혜성 질량**: 혜성의 초기 질량 (일반적으로 10¹²kg)
    - **이심률**: 0=원궤도, 1에 가까울수록 타원궤도
    - **긴반지름**: 궤도 타원의 가장 긴 반지름
    - **질량 소실률**: 혜성이 초당 잃는 질량
    
    **물리학적 기반:**
    - 케플러 궤도역학 사용
    - 질량 소실이 궤도 안정성에 미치는 영향 모델링
    - 중력은 항성과 혜성 간의 이체 문제로 단순화
    """)
    
    # 정보 섹션
    st.markdown("---")
    st.markdown("### ℹ️ 시뮬레이션 정보")
    st.markdown("""
    이 시뮬레이션은 혜성의 질량 소실이 궤도에 미치는 영향을 보여줍니다.
    실제 혜성은 태양에 가까워질 때 얼음이 승화되어 질량을 잃으며, 
    이는 궤도의 이심률과 안정성에 영향을 미칩니다.
    
    **제한사항:**
    - 이체 문제로 단순화 (다른 행성의 영향 무시)
    - 상대론적 효과 무시
    - 비등방적 질량 소실 효과 단순화
    """)

if __name__ == "__main__":
    main()
