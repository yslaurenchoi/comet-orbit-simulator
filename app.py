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
        - eccentricity: 이심률 (0-2: 0=원, 0<e<1=타원, e=1=포물선, e>1=쌍곡선)
        - semi_major_axis: 긴반지름 (AU) - 타원 궤도일 때만 적용
        - mass_loss_rate: 질량 소실률 (kg/s)
        """
        self.star_mass = star_mass * M_sun
        self.initial_comet_mass = comet_mass
        self.current_comet_mass = comet_mass
        self.eccentricity = eccentricity
        self.semi_major_axis = semi_major_axis * AU
        self.mass_loss_rate = mass_loss_rate
        self.is_extinct = False
        self.extinction_time = None
        
        # 궤도 타입 결정
        if eccentricity < 1.0:
            self.orbit_type = "타원"
            # 케플러 제3법칙으로 궤도 주기 계산
            self.orbital_period = 2 * np.pi * np.sqrt(self.semi_major_axis**3 / (G * self.star_mass))
        elif eccentricity == 1.0:
            self.orbit_type = "포물선"
            self.orbital_period = np.inf  # 무한대
        else:
            self.orbit_type = "쌍곡선"
            self.orbital_period = np.inf  # 무한대
    
    def get_orbital_position(self, time):
        """주어진 시간에서의 궤도 위치 계산 (수학적으로 정확한 케플러 궤도)"""
        if self.is_extinct:
            return None, None, None
        
        if self.eccentricity < 1.0:
            # 타원 궤도
            return self._elliptical_orbit_position(time)
        elif self.eccentricity == 1.0:
            # 포물선 궤도
            return self._parabolic_orbit_position(time)
        else:
            # 쌍곡선 궤도
            return self._hyperbolic_orbit_position(time)
    
    def _elliptical_orbit_position(self, time):
        """타원 궤도 위치 계산"""
        # 평균 근점 이상
        mean_anomaly = 2 * np.pi * time / self.orbital_period
        
        # 이심 근점 이상 (뉴턴 방법으로 해결)
        eccentric_anomaly = self._solve_kepler_equation(mean_anomaly, self.eccentricity)
        
        # 참 근점 이상
        true_anomaly = 2 * np.arctan(np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity)) * 
                                     np.tan(eccentric_anomaly / 2))
        
        # 궤도 반지름
        r = self.semi_major_axis * (1 - self.eccentricity**2) / (1 + self.eccentricity * np.cos(true_anomaly))
        
        # 직교 좌표계로 변환
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        return x, y, r
    
    def _parabolic_orbit_position(self, time):
        """포물선 궤도 위치 계산"""
        # 포물선 궤도의 경우 근일점 거리를 사용
        periapsis_distance = self.semi_major_axis * AU  # AU를 미터로 변환
        
        # 평균 운동 계산
        n = np.sqrt(G * self.star_mass / (2 * periapsis_distance**3))
        
        # 바커 방정식 해결 (근사해)
        M = n * time
        D = np.cbrt(3 * M + np.sqrt(9 * M**2 + 8))
        true_anomaly = 2 * np.arctan(D - 2/D)
        
        # 궤도 반지름
        r = periapsis_distance * (1 + np.cos(true_anomaly))
        
        # 직교 좌표계로 변환
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        return x, y, r
    
    def _hyperbolic_orbit_position(self, time):
        """쌍곡선 궤도 위치 계산"""
        # 쌍곡선 궤도의 경우 근일점 거리 사용
        periapsis_distance = self.semi_major_axis * AU
        
        # 평균 운동 계산 (쌍곡선 궤도)
        n = np.sqrt(G * self.star_mass / (-self.semi_major_axis * AU)**3)
        
        # 평균 근점 이상
        mean_anomaly = n * time
        
        # 쌍곡선 케플러 방정식 해결
        hyperbolic_anomaly = self._solve_hyperbolic_kepler_equation(mean_anomaly, self.eccentricity)
        
        # 참 근점 이상
        true_anomaly = 2 * np.arctan(np.sqrt((self.eccentricity + 1) / (self.eccentricity - 1)) * 
                                     np.tanh(hyperbolic_anomaly / 2))
        
        # 궤도 반지름
        r = abs(self.semi_major_axis) * AU * (self.eccentricity**2 - 1) / (1 + self.eccentricity * np.cos(true_anomaly))
        
        # 직교 좌표계로 변환
        x = r * np.cos(true_anomaly)
        y = r * np.sin(true_anomaly)
        
        return x, y, r
    
    def _solve_kepler_equation(self, mean_anomaly, eccentricity, tolerance=1e-10):
        """타원 궤도 케플러 방정식을 뉴턴 방법으로 해결"""
        eccentric_anomaly = mean_anomaly
        
        for _ in range(100):
            f = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly
            if abs(f) < tolerance:
                break
            df = 1 - eccentricity * np.cos(eccentric_anomaly)
            eccentric_anomaly = eccentric_anomaly - f / df
        
        return eccentric_anomaly
    
    def _solve_hyperbolic_kepler_equation(self, mean_anomaly, eccentricity, tolerance=1e-10):
        """쌍곡선 궤도 케플러 방정식을 뉴턴 방법으로 해결"""
        hyperbolic_anomaly = mean_anomaly
        
        for _ in range(100):
            f = eccentricity * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly - mean_anomaly
            if abs(f) < tolerance:
                break
            df = eccentricity * np.cosh(hyperbolic_anomaly) - 1
            hyperbolic_anomaly = hyperbolic_anomaly - f / df
        
        return hyperbolic_anomaly
    
    def update_mass(self, time_step):
        """질량만 업데이트 (궤도에는 영향 없음)"""
        if self.is_extinct:
            return
        
        mass_loss = self.mass_loss_rate * time_step
        
        if self.current_comet_mass - mass_loss <= 0:
            self.current_comet_mass = 0
            self.is_extinct = True
            return
        
        self.current_comet_mass -= mass_loss
    
    def generate_orbit_data(self, total_time, time_steps):
        """전체 궤도 데이터 생성"""
        times = np.linspace(0, total_time, time_steps)
        positions = []
        masses = []
        
        for i, t in enumerate(times):
            if i > 0:
                time_step = times[i] - times[i-1]
                self.update_mass(time_step)
            
            if self.is_extinct:
                break
            
            x, y, r = self.get_orbital_position(t)
            
            if x is not None:
                positions.append((x, y))
                masses.append(self.current_comet_mass)
            else:
                break
        
        actual_times = times[:len(positions)]
        return actual_times, positions, masses

def main():
    st.title("🌟 혜성 궤도 시뮬레이터")
    st.markdown("혜성의 케플러 궤도와 질량 소실을 시뮬레이션합니다.")
    
    # 사이드바 매개변수
    st.sidebar.header("🔧 시뮬레이션 매개변수")
    
    # 항성 질량
    star_mass = st.sidebar.slider(
        "항성 질량 (태양질량 단위)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="항성의 질량을 태양질량 단위로 입력하세요."
    )
    
    # 혜성 초기 질량
    comet_mass_exp = st.sidebar.slider(
        "혜성 초기 질량 (10^x kg)",
        min_value=10,
        max_value=15,
        value=12,
        step=1,
        help="혜성의 초기 질량을 10의 거듭제곱으로 설정하세요."
    )
    comet_mass = 10**comet_mass_exp
    
    # 궤도 이심률 (0-2)
    eccentricity = st.sidebar.slider(
        "궤도 이심률",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.01,
        help="0=원궤도, 0<e<1=타원, e=1=포물선, e>1=쌍곡선"
    )
    
    # 이심률에 따른 궤도 타입 표시
    if eccentricity < 1.0:
        orbit_type = "타원 궤도"
        orbit_color = "green"
    elif eccentricity == 1.0:
        orbit_type = "포물선 궤도"
        orbit_color = "orange"
    else:
        orbit_type = "쌍곡선 궤도"
        orbit_color = "red"
    
    st.sidebar.markdown(f"**궤도 타입:** :{orbit_color}[{orbit_type}]")
    
    # 긴반지름/근일점 거리
    if eccentricity < 1.0:
        distance_label = "긴반지름 (AU)"
        distance_help = "타원 궤도의 긴반지름"
    else:
        distance_label = "근일점 거리 (AU)"
        distance_help = "포물선/쌍곡선 궤도의 근일점 거리"
    
    semi_major_axis = st.sidebar.slider(
        distance_label,
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help=distance_help
    )
    
    # 질량 소실률 (더 작은 범위)
    mass_loss_exp = st.sidebar.slider(
        "질량 소실률 (10^x kg/s)",
        min_value=1,
        max_value=6,
        value=4,
        step=1,
        help="혜성이 초당 잃는 질량 (더 작은 범위)"
    )
    mass_loss_rate = 10**mass_loss_exp
    
    # 예상 생존시간 계산 및 자동 시뮬레이션 기간 설정
    estimated_lifetime_years = comet_mass / mass_loss_rate / YEAR
    auto_sim_years = min(max(estimated_lifetime_years * 1.2, 1), 200)  # 생존시간의 120%, 최소 1년, 최대 200년
    
    st.sidebar.markdown(f"### 🔮 예상 혜성 생존시간: {estimated_lifetime_years:.1f}년")
    st.sidebar.markdown(f"### ⏱️ 자동 설정 시뮬레이션 기간: {auto_sim_years:.1f}년")
    
    if estimated_lifetime_years < auto_sim_years * 0.8:
        st.sidebar.warning(f"⚠️ 혜성이 {estimated_lifetime_years:.1f}년 후 완전히 소멸됩니다!")
    
    # 현재 설정 표시
    st.sidebar.markdown("### 📊 현재 설정값")
    st.sidebar.write(f"**항성 질량:** {star_mass:.1f} 태양질량")
    st.sidebar.write(f"**혜성 질량:** {comet_mass:.1e} kg")
    st.sidebar.write(f"**이심률:** {eccentricity:.2f} ({orbit_type})")
    st.sidebar.write(f"**{distance_label.split('(')[0].strip()}:** {semi_major_axis:.1f} AU")
    st.sidebar.write(f"**질량소실률:** {mass_loss_rate:.1e} kg/s")
    
    # 시뮬레이션 실행
    if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
        simulator = CometOrbitSimulator(
            star_mass=star_mass,
            comet_mass=comet_mass,
            eccentricity=eccentricity,
            semi_major_axis=semi_major_axis,
            mass_loss_rate=mass_loss_rate
        )
        
        total_time = auto_sim_years * YEAR
        time_steps = 1000
        
        with st.spinner("시뮬레이션 계산 중..."):
            times, positions, masses = simulator.generate_orbit_data(total_time, time_steps)
        
        if simulator.is_extinct:
            extinction_time_years = len(positions) * auto_sim_years / time_steps
            st.warning(f"🔥 **혜성이 {extinction_time_years:.1f}년 후 완전히 소멸되었습니다!**")
        
        # 결과 표시
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🌍 궤도 애니메이션")
            
            fig = go.Figure()
            
            # 항성
            fig.add_trace(go.Scatter(
                x=[0], y=[0],
                mode='markers',
                marker=dict(size=25, color='gold', symbol='star'),
                name='항성',
                hovertemplate='<b>항성</b><br>질량: %.1f 태양질량<extra></extra>' % star_mass
            ))
            
            # 궤도 경로
            x_pos = [pos[0]/AU for pos in positions]
            y_pos = [pos[1]/AU for pos in positions]
            
            fig.add_trace(go.Scatter(
                x=x_pos, y=y_pos,
                mode='lines',
                line=dict(color='lightblue', width=2),
                name=f'{orbit_type} 경로',
                hovertemplate=f'{orbit_type} 경로<extra></extra>'
            ))
            
            # 애니메이션 프레임 생성
            frames = []
            for i in range(0, len(positions), max(1, len(positions)//100)):
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
                if masses[i] > 0:
                    comet_size = max(8, 20 * masses[i] / comet_mass)
                    comet_color = 'red' if masses[i] > comet_mass * 0.1 else 'orange'
                    
                    frame_data.append(go.Scatter(
                        x=[x_pos[i]], y=[y_pos[i]],
                        mode='markers',
                        marker=dict(size=comet_size, color=comet_color, symbol='circle'),
                        name='혜성',
                        hovertemplate=f'<b>혜성</b><br>시간: {times[i]/YEAR:.1f}년<br>질량: {masses[i]:.2e} kg<extra></extra>'
                    ))
                else:
                    frame_data.append(go.Scatter(
                        x=[x_pos[i]], y=[y_pos[i]],
                        mode='markers',
                        marker=dict(size=15, color='gray', symbol='x'),
                        name='소멸된 혜성',
                        hovertemplate=f'<b>혜성 소멸</b><br>시간: {times[i]/YEAR:.1f}년<extra></extra>'
                    ))
                
                frames.append(go.Frame(data=frame_data, name=str(i)))
            
            fig.frames = frames
            
            # 레이아웃 설정
            max_distance = max(max(abs(x) for x in x_pos), max(abs(y) for y in y_pos)) * 1.2
            
            fig.update_layout(
                title=f"혜성 {orbit_type} 시뮬레이션 (e={eccentricity:.2f})",
                xaxis_title="거리 (AU)",
                yaxis_title="거리 (AU)",
                showlegend=True,
                width=700,
                height=600,
                xaxis=dict(
                    scaleanchor="y", 
                    scaleratio=1,
                    range=[-max_distance, max_distance]
                ),
                yaxis=dict(
                    range=[-max_distance, max_distance]
                ),
                plot_bgcolor='rgba(0,0,0,0.05)',
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'x': 0.1,
                    'y': 0,
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
            st.subheader("📈 질량 변화")
            
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
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_mass, use_container_width=True)
            
            # 궤도 정보
            st.subheader("📊 궤도 정보")
            st.markdown(f"""
            **궤도 타입:** {orbit_type}
            **이심률:** {eccentricity:.3f}
            **궤도 분류:**
            - 0.0: 완전한 원궤도
            - 0.0 < e < 1.0: 타원궤도
            - e = 1.0: 포물선궤도 (탈출 궤도)
            - e > 1.0: 쌍곡선궤도 (탈출 궤도)
            """)
        
        # 결과 요약
        st.subheader("📊 시뮬레이션 결과 요약")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "궤도 이심률",
                f"{eccentricity:.3f}"
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
    
    # 도움말
    st.sidebar.markdown("---")
    st.sidebar.subheader("📚 도움말")
    st.sidebar.markdown("""
    **궤도 이심률 가이드:**
    - **0.0**: 완전한 원궤도
    - **0.0 < e < 1.0**: 타원궤도 (행성/혜성)
    - **e = 1.0**: 포물선궤도 (탈출속도)
    - **e > 1.0**: 쌍곡선궤도 (항성간 천체)
    
    **수학적 정확성:**
    - 케플러의 궤도역학 법칙 적용
    - 뉴턴 방법으로 궤도 방정식 해결
    - 타원/포물선/쌍곡선 궤도 모두 지원
    """)
    
    # 정보
    st.markdown("---")
    st.markdown("### ℹ️ 시뮬레이션 정보")
    st.markdown("""
    **수학적으로 정확한 케플러 궤도:**
    - 타원 궤도 (e < 1): 케플러 방정식 사용
    - 포물선 궤도 (e = 1): 바커 방정식 사용  
    - 쌍곡선 궤도 (e > 1): 쌍곡선 케플러 방정식 사용
    
    **물리적 특징:**
    - 질량 소실은 궤도에 영향을 주지 않음 (실제로도 미미함)
    - 궤도는 초기 조건에만 의존
    - 혜성은 질량이 0이 되면 소멸
    """)

if __name__ == "__main__":
    main()
