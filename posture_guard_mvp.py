import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd

st.set_page_config(page_title="PosteGuard AI | Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap');
    .stApp { background-color: #081627 !important; color: #FFFFFF !important; }
    .header-container { display: flex; align-items: center; padding: 20px; background: rgba(0, 212, 255, 0.05); border-radius: 15px; border: 1px solid #00D4FF; margin-bottom: 25px; }
    h1 { font-family: 'Orbitron', sans-serif !important; color: #00D4FF !important; text-shadow: 0 0 15px rgba(0, 212, 255, 0.4); margin: 0 !important; }
    .metric-card { background: #0B1E33; padding: 20px; border-radius: 12px; border-left: 5px solid #00D4FF; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
    .metric-val { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; color: #00D4FF; }
    .medical-panel { background: rgba(255, 255, 255, 0.02); padding: 20px; border-radius: 15px; border: 1px dashed #4A6A8A; }
    .stCheckbox { background: #00D4FF !important; border-radius: 8px !important; padding: 10px 20px !important; }
    .stCheckbox label p { color: #081627 !important; font-weight: 800 !important; text-transform: uppercase; }
    .stImage > img { border: 3px solid #00D4FF !important; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header-container"><h1>🛡️ POSTEGUARD <span style="font-size:1rem; font-family:Inter; color:#4A6A8A; font-weight:400;">| ANÁLISE TERMOLÓGICA</span></h1></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    background_mode = st.toggle("Executar em Segundo Plano", value=True)
    st.markdown("---")
    st.markdown("### 👨‍⚕️ Acesso Médico")
    med_auth = st.text_input("Token", "PG-8821-X", type="password")

col_left, col_right = st.columns([1.2, 2], gap="large")

with col_left:
    st.markdown("### 📊 Análises em tempo real")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown('<div class="metric-card"><small>TENSÃO ATUAL</small><br><span class="metric-val">12%</span></div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="metric-card"><small>PONTUAÇÃO ERGO</small><br><span class="metric-val" style="color:#00FFAB">9,4</span></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="medical-panel">', unsafe_allow_html=True)
    st.write("🩺 **Relatório para Especialista**")
    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=['Cervical', 'Lombar', 'Ombros'])
    st.line_chart(chart_data, height=150)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.checkbox('SCANNER ATIVAR POSTEGUARD', value=True)

with col_right:
    mode = st.radio("Modo de Visualização:", ["Câmeras", "Ondas de Calor", "Mapeamento Ósseo"], horizontal=True)
    FRAME_WINDOW = st.image([])
    
    st.markdown("### 🌡️ Mapeamento Térmico Corporal")
    # AQUI ESTÁ A CORREÇÃO DO CAMINHO:
    st.image("imagem/images.jpg", use_container_width=True, caption="Detecção de pontos de calor (Trigger Points)")

if run:
    try:
        import mediapipe as mp
        # AQUI ESTÁ A CORREÇÃO DO MEDIAPIPE (solutions):
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(min_detection_confidence=0.6)
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(rgb)
            
            if mode == "Câmeras":
                display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif mode == "Ondas de Calor":
                intensity_map = np.zeros((h, w), dtype=np.uint8)
                if results.pose_landmarks:
                    for pt in [results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12]]:
                        cv2.circle(intensity_map, (int(pt.x*w), int(pt.y*h)), 150, 255, -1)
                intensity_map = cv2.GaussianBlur(intensity_map, (95, 95), 0)
                heatmap = cv2.applyColorMap(intensity_map, cv2.COLORMAP_JET)
                display_frame = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            elif mode == "Mapeamento Ósseo":
                mp_drawing = mp.solutions.drawing_utils
                display_frame = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(display_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            FRAME_WINDOW.image(display_frame)
            if not background_mode: time.sleep(0.1)
        cap.release()
    except Exception as e:
        st.error(f"Erro no carregamento: {e}")