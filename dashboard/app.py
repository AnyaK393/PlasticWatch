"""PlasticWatch PS-08: visual decision support, never autonomous enforcement."""
from pathlib import Path
import sys
import altair as alt
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.plasticwatch_pipeline import run_plasticwatch
from backend.taco_adapter import dataset_status

st.set_page_config("PlasticWatch | PS-08", "♻️", layout="wide")
st.markdown("""<style>.stApp{background:#07151c}.hero{padding:1.25rem;border-radius:18px;background:linear-gradient(115deg,#073d48,#172f25);border:1px solid #2d7568}.card{padding:.9rem 1rem;border:1px solid #29525a;border-radius:12px;background:#0d2229;min-height:104px}.tiny{color:#b4ced0;font-size:.82rem}.step{text-align:center;padding:.65rem .25rem;border-radius:10px;background:#113039;border:1px solid #2b6365;min-height:80px}</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data(): return run_plasticwatch()

def colour(priority): return {"CRITICAL":"#ef4444", "HIGH":"#f59e0b", "VERIFY":"#38bdf8"}[priority]
def get_hotspot(items, identifier): return next(item for item in items if item["id"] == identifier)
def card(label, value, note, accent="#7dd3fc"):
    st.markdown(f"<div class='card'><span class='tiny'>{label}</span><h2 style='margin:.22rem 0;color:{accent}'>{value}</h2><span class='tiny'>{note}</span></div>", unsafe_allow_html=True)

def map_view(items, selected):
    fmap = folium.Map([18.5204,73.8567], zoom_start=12, tiles="CartoDB dark_matter", control_scale=True)
    for item in items:
        c = colour(item["priority"])
        folium.Circle([item["lat"],item["lon"]], radius=180, color=c, weight=1, fill=True, fill_opacity=.06, tooltip="180 m duplicate-report merge radius").add_to(fmap)
        folium.CircleMarker([item["lat"],item["lon"]], radius=14 if item["id"] == selected else 9, color="#fff" if item["id"] == selected else c, weight=2, fill=True, fill_color=c, fill_opacity=.95, tooltip=f"{item['id']} · {item['priority']} · {item['score']}/100 · drain {item['nearest_drain_m']} m").add_to(fmap)
    legend = """<div style='position:fixed;bottom:28px;left:28px;z-index:9999;background:#0d2229;color:white;padding:10px;border:1px solid #355861;border-radius:8px;font-size:12px'><b>Priority</b><br><span style='color:#ef4444'>●</span> Critical &nbsp; <span style='color:#f59e0b'>●</span> High &nbsp; <span style='color:#38bdf8'>●</span> Verify<br><small>Halo = duplicate merge radius</small></div>"""
    fmap.get_root().html.add_child(folium.Element(legend)); return fmap

data = load_data(); hotspots = data["hotspots"]; reports = [r for h in hotspots for r in h["reports"]]; taco = dataset_status()
count, verified = len(reports), sum(r["verification"] == "verified" for r in reports)
st.sidebar.title("♻️ PlasticWatch"); st.sidebar.caption("PS-08 · Climate, Water & Circularity"); st.sidebar.success("DEMO MODE · transparent simulation replay")
st.sidebar.caption("The map uses open basemap tiles; no map API key is required."); st.sidebar.markdown("---"); st.sidebar.markdown("**Data boundary**"); st.sidebar.caption("Six synthetic, TACO-style report records drive this demonstration. They are not real incidents or cleanup orders."); st.sidebar.markdown("**Decision boundary**"); st.sidebar.caption("AI flags probable waste. A person must verify material and site conditions before cleanup action; the system never assigns responsibility.")
st.markdown(f"<div class='hero'><div style='color:#7dd3fc;font-weight:700'>PLASTICWATCH / PS-08 · CLIMATE, WATER & CIRCULARITY · <span style='color:#fbbf24'>DEMO</span></div><h1 style='margin:.3rem 0'>Plastic Waste Intelligence Command Center</h1><div class='tiny'>{data['area']} · {data['mode_label']} · generated {data['generated_at']}</div></div>", unsafe_allow_html=True)
st.write("")
k1,k2,k3,k4,k5 = st.columns(5)
with k1: card("ACTIVE HOTSPOTS",str(len(hotspots)),"merged by 180 m spatial rule")
with k2: card("REPORTS RECEIVED",str(count),"simulation records in this run")
with k3: card("PLASTIC EVIDENCE",f"{sum(r['plastic_confidence'] for r in reports)/count:.0%}","mean reported plastic confidence")
with k4: card("DRAIN-RISK SITES",str(sum(h["nearest_drain_m"] < 100 for h in hotspots)),"hotspots within 100 m of a drain","#fbbf24")
with k5: card("VERIFICATION RATE",f"{verified/count:.0%}",f"{verified} of {count} reports field-verified","#86efac")

overview,evidence,operations,model_lab = st.tabs(["🗺️ Situation map","🔎 Evidence & explainability","✅ Verification operations","🧠 Model readiness"])
with overview:
    left,right = st.columns([2.5,1])
    with right:
        selected = st.selectbox("Selected hotspot",[h["id"] for h in hotspots]); h = get_hotspot(hotspots,selected)
        card(h["priority"],f"{h['score']}/100","Priority score · evidence + severity + recurrence + drain exposure",colour(h["priority"])); st.write(""); st.metric("Nearest mapped drain",f"{h['nearest_drain_m']} m"); st.metric("Merged reports",h["recurrence"]); st.caption("Drain distance is part of the replay record, not a live water-network feed.")
        if st.button("Mark for field verification",type="primary",width="stretch"): st.toast(f"{selected} marked in this demo session. No external work order was created.")
    with left: st_folium(map_view(hotspots,selected),height=535,use_container_width=True,returned_objects=[])
    rank = pd.DataFrame(hotspots)[["id","score","priority","recurrence","nearest_drain_m"]].rename(columns={"id":"Hotspot","score":"Priority score","priority":"Band","recurrence":"Reports","nearest_drain_m":"Drain (m)"})
    fig = alt.Chart(rank, title="Priority ranking · verification order").mark_bar().encode(x=alt.X("Priority score:Q",scale=alt.Scale(domain=[0,100])),y=alt.Y("Hotspot:N",sort="-x"),color=alt.Color("Band:N",scale=alt.Scale(domain=["CRITICAL","HIGH","VERIFY"],range=["#ef4444","#f59e0b","#38bdf8"])),tooltip=["Hotspot","Priority score","Band","Reports","Drain (m)"]).properties(height=220)
    st.altair_chart(fig,width="stretch")
with evidence:
    selected = st.selectbox("Evidence for hotspot",[h["id"] for h in hotspots],key="evidence"); h = get_hotspot(hotspots,selected); e1,e2 = st.columns([1.05,1])
    with e1:
        st.subheader(f"Why {h['id']} is {h['priority']}")
        for label,points,maximum in [("Plastic evidence",h["confidence"]*35,35),("Severity",h["severity"]*8,40),("Repeat reports",min(h["recurrence"],4)*7,28),("Drain exposure",max(0,20-h["nearest_drain_m"]/10),20)]:
            st.markdown(f"**{label}** · {points:.1f}/{maximum}"); st.progress(min(int(points/maximum*100),100),text=f"{points:.1f} score points")
        with st.expander("Show scoring rule and safeguards"):
            st.code(data["formula"]); st.caption("The score prioritises field attention; it is not proof of waste type, ownership, fault, or a directive to clean up.")
    with e2:
        st.subheader("Waste situation"); severity = pd.DataFrame(reports).groupby("severity").size().reset_index(name="reports"); severity["severity"] = severity["severity"].astype(str)
        fig = alt.Chart(severity, title="Severity distribution").mark_bar(color="#38bdf8").encode(x=alt.X("severity:N",title="Reported severity (1–5)"),y=alt.Y("reports:Q",title="Reports"),tooltip=["severity","reports"]).properties(height=205); st.altair_chart(fig,width="stretch"); st.info("Waste composition and time trends are intentionally not shown: replay data has neither detector class labels nor timestamps.")
    st.subheader("Evidence cards"); columns = st.columns(len(h["reports"]))
    for column,r in zip(columns,h["reports"]):
        with column: card(r["id"],f"{r['plastic_confidence']:.0%}",f"severity {r['severity']} · {r['near_drain_m']} m to drain<br>{r['verification']}","#86efac" if r["verification"] == "verified" else "#7dd3fc")
    with st.expander("Open raw evidence records"): st.dataframe(pd.DataFrame(h["reports"])[["id","plastic_confidence","severity","near_drain_m","source","verification"]],width="stretch",hide_index=True)
with operations:
    st.subheader("Human verification loop"); cols = st.columns(5)
    for column,step in zip(cols,["1. Report<br>photo + location","2. AI triage<br>probability only","3. Merge<br>near-duplicate reports","4. Verify<br>field worker checks","5. Record<br>before/after evidence"]):
        with column: st.markdown(f"<div class='step'>{step}</div>",unsafe_allow_html=True)
    pending = [h for h in hotspots if h["verified_count"] < h["recurrence"]]; queue = pd.DataFrame([{"Order":i+1,"Hotspot":h["id"],"Priority":h["priority"],"Why now":f"{h['recurrence']} reports · drain {h['nearest_drain_m']} m","Field status":f"{h['verified_count']}/{h['recurrence']} reports verified"} for i,h in enumerate(pending)])
    st.write(""); st.dataframe(queue,width="stretch",hide_index=True); c1,c2 = st.columns(2)
    with c1: card("VERIFIED",f"{verified} reports","Evidence for review, not automated cleanup completion.","#86efac")
    with c2: card("AWAITING FIELD CHECK",f"{count-verified} reports","Queue ranked by transparent hotspot priority.","#fbbf24")
    st.warning("Before/after verification is a demo workflow only. This replay has no real cleanup event, photo, person, or work order attached.")
    with st.expander("Open verification details"): st.dataframe(pd.DataFrame(reports)[["id","source","verification","near_drain_m"]],width="stretch",hide_index=True)
with model_lab:
    st.subheader("TACO → PlasticWatch detector readiness")
    if not taco["available"]: st.error(taco["message"])
    else:
        m1,m2,m3,m4 = st.columns(4)
        with m1: card("TACO ANNOTATIONS",f"{taco['annotations']:,}",f"across {taco['images']:,} COCO image records")
        with m2: card("TARGET CLASSES",str(len(taco["target_classes"])),"curated classes for this prototype")
        with m3: card("DOWNLOADED IMAGES",str(taco["downloaded_images"]),"actual image files detected","#fbbf24" if not taco["downloaded_images"] else "#86efac")
        with m4: card("MODEL STATE",taco["model_state"].upper(),"no detector outputs are used in this demo","#fbbf24")
        st.caption(taco["license"]); st.markdown("**Target classes**"); class_cols=st.columns(len(taco["target_classes"]))
        for column,target in zip(class_cols,taco["target_classes"]):
            with column: st.markdown(f"<div class='step'>♻️<br><b>{target.replace('_',' ')}</b></div>",unsafe_allow_html=True)
        if taco["downloaded_images"] == 0: st.warning("Preparation and training are correctly blocked: annotations are present, but no TACO image files have been downloaded. The dashboard uses synthetic replay evidence, not model predictions.")
        elif not taco["prepared_dataset"]: st.info("Images are present. Run the reproducible conversion command in the README to create YOLO labels and dataset.yaml.")
        elif not taco["trained_model"]: st.info("YOLO data is prepared but no `best.pt` exists. Train in Python 3.10–3.12 with PyTorch and Ultralytics, then evaluate before connecting inference.")
        else: st.success("A trained model artifact is present. Add evaluated inference integration before using it for live triage.")
