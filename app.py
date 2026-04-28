import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from predict import CrackPredictor, SpecimenData

st.set_page_config(page_title="Crack Prediction System", page_icon="🔩", layout="centered")

st.title("🔩 Structural Crack Prediction System")
st.caption("Physics-based ML using Paris-Erdogan Law · Irwin SIF · Murakami √area Model")

with st.sidebar:
    st.header("📋 Specimen Inputs")
    material = st.selectbox("Material", ["aluminum_2024", "steel_4340", "titanium_6al4v"])
    panel_width = st.number_input("Panel Width (m)", value=0.10, min_value=0.01, max_value=1.0, step=0.01)
    sigma_max = st.number_input("Max Stress σ_max (MPa)", value=150.0, min_value=1.0, max_value=1000.0)
    sigma_min = st.number_input("Min Stress σ_min (MPa)", value=15.0, min_value=0.0, max_value=500.0)
    load_cycles = st.number_input("Load Cycles N", value=80000, min_value=100, max_value=5000000, step=1000)
    crack_length = st.number_input("Crack Length (mm)", value=6.0, min_value=0.1, max_value=100.0)
    ut_amplitude = st.slider("UT Amplitude (NDT signal)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    st.divider()
    st.subheader("🔬 Advanced NDT (optional)")
    tofd_t1 = st.number_input("TOFD t1 (µs)", value=1.5, min_value=0.1, max_value=10.0, step=0.1)
    tofd_tL = st.number_input("TOFD tL (µs)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    tofd_depth = st.number_input("TOFD Depth (mm) — leave 0 for auto", value=0.0, min_value=0.0, max_value=100.0, step=0.1)

    predict_btn = st.button("🔍 Run Prediction", use_container_width=True)

RISK_COLORS = {"SAFE": "green", "LOW": "blue", "MODERATE": "orange", "HIGH": "red", "CRITICAL": "red"}
STAGE_EMOJI = {0: "✅", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🚨"}

if predict_btn:
    with st.spinner("Running prediction..."):
        try:
            predictor = CrackPredictor()
            specimen = SpecimenData(
                material=material,
                panel_width_m=panel_width,
                sigma_max_MPa=sigma_max,
                sigma_min_MPa=sigma_min,
                load_cycles_N=int(load_cycles),
                crack_length_mm=crack_length,
                ut_amplitude=ut_amplitude,
                tofd_t1_us=tofd_t1,
                tofd_tL_us=tofd_tL,
                tofd_depth_mm=tofd_depth if tofd_depth > 0 else None,  # None = auto-compute
            )
            result = predictor.predict(specimen)

            risk = result["risk_level"]
            color = RISK_COLORS.get(risk, "gray")
            emoji = STAGE_EMOJI.get(result["crack_stage"], "❓")

            st.markdown(f"## {emoji} Risk Level: :{color}[**{risk}**]")
            st.divider()

            col1, col2, col3 = st.columns(3)
            col1.metric("Crack Present", "YES ⚠️" if result["crack_present"] else "NO ✅")
            col2.metric("Crack Probability", f"{result['crack_probability']*100:.1f}%")
            col3.metric("Stage", f"{result['crack_stage']} — {result['crack_stage_label']}")

            col4, col5 = st.columns(2)
            col4.metric("Predicted Crack Length", f"{result['predicted_crack_length_mm']:.3f} mm")
            col5.metric("Remaining Useful Life", f"{result['remaining_useful_life_cycles']:,} cycles")

            st.info(f"💡 **Action Required:** {result['recommendation']}")

            with st.expander("🔬 Physics Features"):
                for k, v in result["physics_features"].items():
                    st.write(f"**{k}:** {v}")

            with st.expander("📊 Stage Probabilities"):
                for stage, prob in result["crack_stage_probabilities"].items():
                    st.progress(prob, text=f"Stage {stage}: {prob*100:.1f}%")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("👈 Fill in specimen details in the sidebar and click **Run Prediction**.")
    st.markdown("""
    **What this system predicts:**
    | Task | Output |
    |------|--------|
    | Crack Detection | Crack present / absent |
    | Damage Stage | None → Initiation → Stable → Accelerated → Critical |
    | Crack Length | Predicted length in mm |
    | Remaining Useful Life | Cycles to failure |
    """)