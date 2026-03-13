import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time

# ─────────────────────────────────────────────────────────────────────────────
#  DONNÉES RECYCLAGE
# ─────────────────────────────────────────────────────────────────────────────
RECYCLING_INFO = {
    "battery": {
        "recyclable": True,
        "icon": "🔋",
        "color": "#e74c3c",
        "bin": "Point de collecte spécial",
        "tips": "Ne jamais jeter à la poubelle ! Déposer dans les points de collecte dédiés (supermarchés, déchetteries)."
    },
    "biological": {
        "recyclable": True,
        "icon": "🥬",
        "color": "#27ae60",
        "bin": "Bac marron / compost",
        "tips": "Déchets organiques compostables. Épluchures, restes alimentaires, marc de café."
    },
    "cardboard": {
        "recyclable": True,
        "icon": "📦",
        "color": "#3498db",
        "bin": "Bac bleu",
        "tips": "Aplatir les cartons avant de les recycler. Retirer les rubans adhésifs et agrafes."
    },
    "clothes": {
        "recyclable": True,
        "icon": "👕",
        "color": "#9b59b6",
        "bin": "Bornes textiles",
        "tips": "Déposer dans les bornes de collecte textiles. Même abîmés, les vêtements sont recyclables ou réutilisables."
    },
    "glass": {
        "recyclable": True,
        "icon": "🍾",
        "color": "#27ae60",
        "bin": "Bac vert",
        "tips": "Retirer les bouchons et couvercles. Rincer les contenants."
    },
    "metal": {
        "recyclable": True,
        "icon": "🥫",
        "color": "#f39c12",
        "bin": "Bac jaune",
        "tips": "Vider et rincer les canettes et boîtes de conserve. Papier aluminium accepté."
    },
    "paper": {
        "recyclable": True,
        "icon": "📄",
        "color": "#3498db",
        "bin": "Bac bleu",
        "tips": "Le papier souillé (gras, humide) n'est pas recyclable. Retirer les agrafes et trombones."
    },
    "plastic": {
        "recyclable": True,
        "icon": "🧴",
        "color": "#f39c12",
        "bin": "Bac jaune",
        "tips": "Vérifier le symbole de recyclage (1-7). Vider et rincer. Retirer les bouchons."
    },
    "shoes": {
        "recyclable": True,
        "icon": "👟",
        "color": "#e67e22",
        "bin": "Bornes de collecte spécialisées",
        "tips": "Attacher les chaussures par paires. Les déposer dans les bornes dédiées ou associations."
    },
    "trash": {
        "recyclable": False,
        "icon": "🗑️",
        "color": "#95a5a6",
        "bin": "Bac noir / gris",
        "tips": "Déchets non recyclables. À jeter dans la poubelle ordinaire."
    }
}

# ─────────────────────────────────────────────────────────────────────────────
#  MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
def create_head(in_f, hidden, drop, out_f):
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_f, hidden),
        nn.ReLU(),
        nn.BatchNorm1d(hidden),
        nn.Dropout(p=drop),
        nn.Linear(hidden, out_f)
    )

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

@st.cache_resource
def load_model(model_path, model_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name   = model_config.get("model_name",    "efficientnet_b0")
    hidden_units = model_config.get("hidden_units",  192)
    dropout      = model_config.get("dropout",       0.456)
    num_classes  = model_config.get("num_classes",   10)

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = create_head(model.fc.in_features, hidden_units, dropout, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier = create_head(model.classifier[1].in_features, hidden_units, dropout, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier = create_head(model.classifier[1].in_features, hidden_units, dropout, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = create_head(model.classifier.in_features, hidden_units, dropout, num_classes)
    elif model_name == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(weights=None)
        model.fc = create_head(model.fc.in_features, hidden_units, dropout, num_classes)

    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, device

def predict_with_details(model, image, device, class_names):
    transform   = get_transform()
    start_time  = time.time()
    img_tensor  = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs       = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    inference_time = time.time() - start_time
    probs          = probabilities.cpu().numpy()[0]
    pred_class     = class_names[predicted.item()]
    confidence_score = confidence.item()
    entropy        = -np.sum(probs * np.log(probs + 1e-10))

    return {
        "predicted_class": pred_class,
        "confidence": confidence_score,
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
        "inference_time": inference_time,
        "entropy": entropy,
        "top_3_predictions": sorted(
            [(class_names[i], float(probs[i])) for i in range(len(class_names))],
            key=lambda x: x[1], reverse=True
        )[:3]
    }

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "predictions_history" not in st.session_state:
    st.session_state.predictions_history = []

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classificateur de Déchets IA",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS — DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── RESET & ROOT ──────────────────────────────────────────────────────── */
:root {
    --bg:        #0a0c0f;
    --surface:   #111318;
    --surface2:  #181c23;
    --border:    #22272f;
    --accent:    #00e5b0;
    --accent2:   #0084ff;
    --warn:      #f5a623;
    --danger:    #ff4545;
    --text:      #e8eaed;
    --muted:     #6b7280;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'Syne', sans-serif;
    --radius:    6px;
    --transition: 220ms cubic-bezier(.4,0,.2,1);
}

/* ── GLOBAL ────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}

.stApp { background: var(--bg) !important; }

/* ── HIDE DEFAULT STREAMLIT ELEMENTS ───────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px; }

/* ── HEADER STRIP ──────────────────────────────────────────────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0 2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
}
.app-title {
    font-family: var(--sans);
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.app-title span.accent { color: var(--accent); }
.app-badge {
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    background: var(--surface2);
    border: 1px solid var(--border);
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
}

/* ── SECTION LABEL ─────────────────────────────────────────────────────── */
.section-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── UPLOAD ZONE ───────────────────────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color var(--transition), background var(--transition) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: #0a1a14 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* ── IMAGE PREVIEW ─────────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
    object-fit: cover;
}

/* ── BUTTON ────────────────────────────────────────────────────────────── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity var(--transition), transform var(--transition) !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── SPINNER ───────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ── SUCCESS TOAST ─────────────────────────────────────────────────────── */
.stSuccess {
    background: #0a1a14 !important;
    border: 1px solid var(--accent) !important;
    border-radius: var(--radius) !important;
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
}

/* ── ERROR / INFO ──────────────────────────────────────────────────────── */
.stError {
    background: #1a0a0a !important;
    border: 1px solid var(--danger) !important;
    border-radius: var(--radius) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}
.stInfo {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
}

/* ── PREDICTION RESULT CARD ────────────────────────────────────────────── */
.result-class {
    font-family: var(--sans);
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0;
    line-height: 1;
}
.result-class-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.35rem;
}

/* ── RECYCLING CARD ────────────────────────────────────────────────────── */
.recycling-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin: 1.25rem 0;
    position: relative;
    overflow: hidden;
}
.recycling-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: var(--accent-color, var(--accent));
    border-radius: var(--radius) 0 0 var(--radius);
}
.rc-status {
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    display: inline-block;
    margin-bottom: 0.75rem;
}
.rc-status.recyclable {
    background: rgba(0,229,176,.12);
    color: var(--accent);
    border: 1px solid rgba(0,229,176,.25);
}
.rc-status.non-recyclable {
    background: rgba(255,69,69,.12);
    color: var(--danger);
    border: 1px solid rgba(255,69,69,.25);
}
.rc-icon { font-size: 2.5rem; margin-bottom: 0.5rem; display: block; }
.rc-field {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    margin-bottom: 0.35rem;
}
.rc-field strong { color: var(--text); font-weight: 500; }
.rc-tip {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid var(--border);
    line-height: 1.6;
}

/* ── METRIC CARDS ──────────────────────────────────────────────────────── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-value {
    font-family: var(--sans);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin: 0.25rem 0;
}
.metric-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
}
.metric-unit {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--muted);
    margin-left: 0.15rem;
}

/* ── PLOTLY CHARTS ─────────────────────────────────────────────────────── */
.js-plotly-plot .plotly { background: transparent !important; }
[data-testid="stPlotlyChart"] { margin-top: 0 !important; }

/* ── DIVIDER ───────────────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── SUBHEADER ─────────────────────────────────────────────────────────── */
h2, h3 {
    font-family: var(--sans) !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
model_path    = "model.pth"
model_name    = "efficientnet_b0"
hidden_units  = 192
dropout       = 0.456
default_classes = ["battery", "biological", "cardboard", "clothes", "glass",
                   "metal", "paper", "plastic", "shoes", "trash"]
device_info   = "GPU · CUDA" if torch.cuda.is_available() else "CPU"
class_names   = [c.strip() for c in ", ".join(default_classes).split(",")]

model_config = {
    "model_name":  model_name,
    "hidden_units": hidden_units,
    "dropout":      dropout,
    "num_classes":  len(class_names)
}

# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
    <div class="app-title">
        ♻ &nbsp;Waste<span class="accent">Scan</span>
    </div>
    <div style="display:flex;gap:.75rem;align-items:center;">
        <span class="app-badge">{model_name}</span>
        <span class="app-badge">{device_info}</span>
        <span class="app-badge">{len(class_names)} classes</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
try:
    with st.spinner("Chargement du modèle…"):
        model, device = load_model(model_path, model_config)
    # Badge succès discret en haut à droite
    st.markdown("""
    <div style="position:fixed;top:1rem;right:1.5rem;z-index:9999;">
        <span style="
            font-family:var(--mono);font-size:.6rem;font-weight:600;
            letter-spacing:.12em;text-transform:uppercase;
            background:#0a1a14;color:#00e5b0;
            border:1px solid rgba(0,229,176,.3);
            padding:.2rem .7rem;border-radius:2px;">
            ● MODEL READY
        </span>
    </div>""", unsafe_allow_html=True)
except FileNotFoundError:
    st.error(f"Modèle introuvable : `{model_path}`")
    st.info("Assurez-vous que le fichier `.pth` est présent dans le répertoire courant.")
    st.stop()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Image source</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Déposez une image de déchet",
    type=["jpg", "jpeg", "png"],
    help="Formats supportés : JPG, JPEG, PNG",
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSE
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    col_img, col_result = st.columns([1, 1], gap="large")

    # ── Aperçu image ────────────────────────────────────────────────────────
    with col_img:
        st.markdown('<div class="section-label">Aperçu</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)

    # ── Panneau résultat ─────────────────────────────────────────────────────
    with col_result:
        st.markdown('<div class="section-label">Analyse</div>', unsafe_allow_html=True)
        btn = st.button("▶  Analyser l'image", use_container_width=True)

        if btn:
            with st.spinner("Inférence en cours…"):
                results = predict_with_details(model, image, device, class_names)
                st.session_state.predictions_history.append(results)

            pred   = results["predicted_class"]
            conf   = results["confidence"]
            t_inf  = results["inference_time"]
            entr   = results["entropy"]
            info   = RECYCLING_INFO.get(pred, {})

            # ── Classe prédite ───────────────────────────────────────────────
            st.markdown(f"""
            <p class="result-class-label">Catégorie détectée</p>
            <p class="result-class">{info.get('icon','')}&nbsp;{pred.capitalize()}</p>
            """, unsafe_allow_html=True)

            # ── Métriques ────────────────────────────────────────────────────
            st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
            mc1, mc2, mc3 = st.columns(3)
            conf_color = "#00e5b0" if conf >= 0.75 else ("#f5a623" if conf >= 0.5 else "#ff4545")

            with mc1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Confiance</div>
                    <div class="metric-value" style="color:{conf_color};">{conf*100:.1f}<span class="metric-unit">%</span></div>
                </div>""", unsafe_allow_html=True)
            with mc2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Inférence</div>
                    <div class="metric-value">{t_inf*1000:.0f}<span class="metric-unit">ms</span></div>
                </div>""", unsafe_allow_html=True)
            with mc3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Entropie</div>
                    <div class="metric-value">{entr:.2f}<span class="metric-unit">H</span></div>
                </div>""", unsafe_allow_html=True)

            # ── Recyclage ────────────────────────────────────────────────────
            if info:
                is_rec    = info["recyclable"]
                rec_cls   = "recyclable" if is_rec else "non-recyclable"
                rec_label = "RECYCLABLE" if is_rec else "NON RECYCLABLE"
                ac        = info.get("color", "#00e5b0")

                st.markdown(f"""
                <div class="recycling-card" style="--accent-color:{ac};">
                    <span class="rc-status {rec_cls}">{rec_label}</span><br>
                    <p class="rc-field"><strong>Destination :</strong>&nbsp;{info['bin']}</p>
                    <p class="rc-tip">💡 {info['tips']}</p>
                </div>""", unsafe_allow_html=True)

            # ── Jauge de confiance ───────────────────────────────────────────
            st.markdown('<div class="section-label" style="margin-top:1.5rem;">Niveau de confiance</div>', unsafe_allow_html=True)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                number={"suffix": "%", "font": {"size": 32, "color": "#e8eaed", "family": "Syne"}},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 0,
                             "tickcolor": "#22272f", "tickfont": {"color": "#6b7280", "size": 10, "family": "JetBrains Mono"}},
                    "bar": {"color": conf_color, "thickness": 0.55},
                    "bgcolor": "#0a0c0f",
                    "borderwidth": 1,
                    "bordercolor": "#22272f",
                    "steps": [
                        {"range": [0, 50],  "color": "rgba(255,69,69,.08)"},
                        {"range": [50, 75], "color": "rgba(245,166,35,.08)"},
                        {"range": [75, 100],"color": "rgba(0,229,176,.08)"}
                    ],
                    "threshold": {"line": {"color": conf_color, "width": 2}, "thickness": 0.9, "value": conf * 100}
                }
            ))
            fig_gauge.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor="#0a0c0f",
                plot_bgcolor="#0a0c0f",
                font={"family": "JetBrains Mono", "color": "#6b7280"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Séparateur ───────────────────────────────────────────────────────────
    if btn:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Distribution des probabilités</div>', unsafe_allow_html=True)

        prob_data = results["probabilities"]
        keys      = list(prob_data.keys())
        vals      = list(prob_data.values())
        pred_idx  = keys.index(pred)

        colors = ["#00e5b0" if i == pred_idx else "#1e2733" for i in range(len(keys))]

        # ── Bar chart ─────────────────────────────────────────────────────────
        fig_bar = go.Figure(go.Bar(
            x=keys,
            y=vals,
            marker=dict(color=colors, line=dict(color="#22272f", width=1)),
            hovertemplate="<b>%{x}</b><br>%{y:.2%}<extra></extra>"
        ))
        fig_bar.update_layout(
            height=320,
            paper_bgcolor="#0a0c0f",
            plot_bgcolor="#111318",
            margin=dict(l=10, r=10, t=10, b=30),
            xaxis=dict(tickfont=dict(family="JetBrains Mono", size=10, color="#6b7280"),
                       gridcolor="#22272f", linecolor="#22272f"),
            yaxis=dict(tickfont=dict(family="JetBrains Mono", size=10, color="#6b7280"),
                       gridcolor="#22272f", linecolor="#22272f",
                       tickformat=".0%"),
            hoverlabel=dict(bgcolor="#181c23", bordercolor="#22272f",
                            font=dict(family="JetBrains Mono", size=11, color="#e8eaed"))
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Pie chart ─────────────────────────────────────────────────────────
        col_pie1, col_pie2 = st.columns([1, 1])
        with col_pie1:
            st.markdown('<div class="section-label">Répartition</div>', unsafe_allow_html=True)
            palette = ["#00e5b0","#0084ff","#f5a623","#ff4545","#9b59b6",
                       "#27ae60","#e67e22","#3498db","#e74c3c","#95a5a6"]
            fig_pie = go.Figure(go.Pie(
                labels=keys,
                values=vals,
                hole=0.55,
                marker=dict(colors=palette[:len(keys)], line=dict(color="#0a0c0f", width=2)),
                textfont=dict(family="JetBrains Mono", size=10, color="#e8eaed"),
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
            ))
            fig_pie.update_layout(
                height=320,
                paper_bgcolor="#0a0c0f",
                plot_bgcolor="#0a0c0f",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(font=dict(family="JetBrains Mono", size=9, color="#6b7280"),
                            bgcolor="#0a0c0f", bordercolor="#22272f"),
                annotations=[dict(text=f"{conf*100:.0f}%", x=.5, y=.5,
                                  font=dict(size=22, family="Syne", color="#e8eaed"), showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Top-3 ─────────────────────────────────────────────────────────────
        with col_pie2:
            st.markdown('<div class="section-label">Top 3 prédictions</div>', unsafe_allow_html=True)
            for rank, (cls, prob) in enumerate(results["top_3_predictions"]):
                bar_w = int(prob * 100)
                icon  = RECYCLING_INFO.get(cls, {}).get("icon", "")
                st.markdown(f"""
                <div style="
                    background:var(--surface);border:1px solid var(--border);
                    border-radius:var(--radius);padding:.85rem 1rem;margin-bottom:.6rem;
                    position:relative;overflow:hidden;">
                    <div style="
                        position:absolute;top:0;left:0;height:100%;width:{bar_w}%;
                        background:rgba(0,229,176,.06);border-right:1px solid rgba(0,229,176,.15);
                        transition:width .4s ease;"></div>
                    <div style="position:relative;display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-family:var(--mono);font-size:.8rem;color:var(--text);">
                            <span style="color:var(--muted);margin-right:.5rem;">#{rank+1}</span>
                            {icon} {cls.capitalize()}
                        </span>
                        <span style="font-family:var(--sans);font-size:1rem;font-weight:700;
                                     color:{'var(--accent)' if rank==0 else 'var(--muted)'};">
                            {prob:.1%}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)
