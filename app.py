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
        - eccentricity: 이심률 (0-1) - 고정값
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
        
        # 궤도 주기 계산 (케플러 제3법칙)
        self.orbital_period = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / (G * self.star_mass))
        
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
        """주어진 시간에서의 궤도 위치 계산 (고정된 궤도)"""
        if self.is_extinct:
            return None, None, None  # 소멸된 혜성은 위치가 없음
        
        # 평균 근점 이상 (Mean Anomaly)
        mean_anomaly = 2 * np.pi * time / self.orbital_period
        
        # 이심 근점 이상 (Eccentric Anomaly) - 뉴턴 방법으로 해결
        eccentric_anomaly = self.solve_kepler_equation(mean_anomaly, self.eccentricity)
        
        # 참 근점 이상 (True Anomaly)
        true_anomaly = 2 * np.arctan(np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)) * 
                                     np.tan(eccentric_anomaly / 2))
        
        # 궤도 반지름 (고정된 궤도 매개변수 사용)
        r = self.semi_major_axis * (1 - self.eccentricity**2) / (1 + self.eccentricity * np.cos(true_anomaly))
        
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
    
    # 궤도 이심률 (고정값)
    eccentricity = st.sidebar.slider(
        "궤도 이심률 (고정)",
        min_value=0.0,
        max_value=0.99,
        value=0.5,
        step=0.01,
        help="0: 완전한 원궤도, 1에 가까울수록 매우 긴 타원궤도 (시뮬레이션 중 변하지 않음)"
    )
    
    # 긴반지름 (AU, 고정값)
    semi_major_axis = st.sidebar.slider(
        "긴반지름 (AU, 고정)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="궤도의 긴반지름을 천문단위(AU)로 입력하세요. (시뮬레이션 중 변하지 않음)"
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
    
    # 혜성 생존 시간 예측
    estimated_lifetime = comet_mass / mass_loss_rate / YEAR
    st.sidebar.markdown(f"### 🔮 예상 혜성 생존시간: {estimated_lifetime:.1f}년")
    
    if estimated_lifetime < sim_years:
        st.sidebar.warning(f"⚠️ 혜성이 {estimated_lifetime:.1f}년 후 완전히 소멸됩니다!")
    
    # 현재 설정 표시
    st.sidebar.markdown("### 📊 현재 설정값")
    st.sidebar.write(f"**항성 질량:** {star_mass:.1f} 태양질량")
    st.sidebar.write(f"**혜성 질량:** {comet_mass:.1e} kg")
    st.sidebar.write(f"**이심률:** {eccentricity:.2f} (고정)")
    st.sidebar.write(f"**긴반지름:** {semi_major_axis:.1f} AU (고정)")
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
                name='궤도 경로 (고정)',
                hovertemplate='고정 궤도 경로<extra></extra>'
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
                        hovertemplate=f'<b>혜성</b><br>시간: {times[i]/YEAR:.1f}년<br>질량: {masses[i]:.2e} kg<br>이심률: {eccentricity:.3f} (고정)<extra></extra>'
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
                title="혜성 궤도 시뮬레이션 (고정 궤도)",
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
            st.subheader("📈 변화 분석")
            
            # 이심률 고정값 표시
            fig_ecc = go.Figure()
            fig_ecc.add_trace(go.Scatter(
                x=[t/YEAR for t in times],
                y=[eccentricity] * len(times),
                mode='lines',
                name='이심률',
                line=dict(color='green', width=3)
            ))
            fig_ecc.update_layout(
                title="이심률 (고정값)",
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
                "궤도 이심률",
                f"{eccentricity:.3f}",
                "고정값"
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
            actual_sim_time = times[-1] / YEAR
            st.metric(
                "실제 시뮬레이션 시간",
                f"{actual_sim_time:.1f} 년"
            )
        
        # 물리학적 해석
        st.subheader("🔬 물리학적 해석")
        
        if simulator.is_extinct:
            interpretation = f"""
            **🔥 혜성 완전 소멸:**
            - 혜성이 {simulator.extinction_time/YEAR:.1f}년 후 완전히 소멸되었습니다.
            - 총 {mass_loss_percent:.1f}%의 질량을 잃고 사라졌습니다.
            - **궤도는 소멸 순간까지 일정하게 유지되었습니다.**
            
            **물리학적 의미:**
            - 질량이 0이 되면 물체가 존재하지 않으므로 궤도 운동도 불가능합니다.
            - 혜성의 질량 소실 자체는 궤도 형태를 변화시키지 않습니다.
            - 실제 혜성에서 궤도 변화는 비등방적 가스 분출에 의한 반작용력 때문입니다.
            """
        else:
            interpretation = f"""
            **질량 소실 과정:**
            - 혜성이 {mass_loss_percent:.1f}%의 질량을 잃었습니다.
            - **궤도는 전혀 변하지 않고 일정하게 유지되었습니다.**
            - 이는 물리학적으로 정확한 결과입니다.
            
            **현재 상태:**
            - 혜성은 아직 존재하며 동일한 궤도에서 운동을 계속합니다.
            - 현재 질량: {masses[-1]:.2e} kg
            - 궤도 이심률: {eccentricity:.3f} (변화 없음)
            """
        
        st.markdown(interpretation)
    
    # 도움말 섹션
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 도움말")
    st.sidebar.markdown("""
    **매개변수 설명:**
    - **항성 질량**: 중심별의 질량 (태양 = 1.0)
    - **혜성 질량**: 혜성의 초기 질량 (일반적으로 10¹²kg)
    - **이심률**: 0=원궤도, 1에 가까울수록 타원궤도 (고정)
    - **긴반지름**: 궤도 타원의 가장 긴 반지름 (고정)
    - **질량 소실률**: 혜성이 초당 잃는 질량
    
    **물리학적 기반:**
    - 케플러 궤도역학 사용
    - 궤도는 질량 소실과 무관하게 일정
    - 질량이 0이 되면 혜성 소멸
    """)
    
    # 정보 섹션
    st.markdown("---")
    st.markdown("### ℹ️ 시뮬레이션 정보")
    st.markdown("""
    이 시뮬레이션은 **물리학적으로 정확한** 혜성의 질량 소실 과정을 보여줍니다.
    
    🔬 **물리학적 특징:**
    - **궤도는 질량 소실과 무관하게 일정합니다** (케플러 법칙)
    - 질량이 0이 되면 혜성이 완전히 소멸됩니다
    - 궤도 변화는 질량 자체가 아닌 **비등방적 가스 분출**에 의해 발생합니다
    
    📝 **수정사항:**
    - 비과학적인 "질량 소실 → 궤도 변화" 로직 제거
    - 궤도 매개변수(이심률, 긴반지름) 고정
    - 물리학적으로 정확한 시뮬레이션 구현
    
    **제한사항:**
    - 이체 문제로 단순화 (다른 행성의 영향 무시)
    - 상대론적 효과 무시
    - 가스 분출에 의한 반작용력 미포함
    """)

if __name__ == "__main__":
    main()
