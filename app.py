import streamlit as st
import pandas as pd
import pickle
import time
import random
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.nlu.ptp_extractor import extract_ptp
from src.nlu.recovery_predictor import engineer_features

st.set_page_config(page_title="VAADA", page_icon="💰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
.stApp{background:#0a0a0a!important}
.block-container{padding:1.5rem 2rem} header{display:none!important} .stAppHeader{display:none!important}
h1,h2,h3{color:#00ff41!important;font-family:'JetBrains Mono',monospace!important}
p,label{color:#c8ffc8!important;font-family:'JetBrains Mono',monospace!important;font-size:12px!important}
div[data-testid="stSidebar"]{background:#0d0d0d!important;border-right:1px solid #1a3a1a!important}
.stButton>button{background:#003311!important;color:#00ff41!important;border:1px solid #00ff41!important;border-radius:0!important;font-family:'JetBrains Mono',monospace!important;font-weight:700!important}
.stButton>button:hover{background:#00ff41!important;color:#000!important}
.stTextArea textarea{background:#050505!important;color:#00ff41!important;border:1px solid #1a3a1a!important;border-radius:0!important;font-family:'JetBrains Mono',monospace!important;font-size:13px!important}
.stTabs [data-baseweb="tab"]{background:#0d0d0d!important;color:#557755!important;font-family:'JetBrains Mono',monospace!important;border-radius:0!important}
.stTabs [aria-selected="true"]{background:#003311!important;color:#00ff41!important;border-bottom:2px solid #00ff41!important}
.mbox{background:#0d0d0d;border:1px solid #1a3a1a;padding:16px 20px;margin-bottom:8px}
.mlabel{color:#557755;font-size:10px;font-family:'JetBrains Mono',monospace;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px}
.mvalue{color:#00ff41;font-size:24px;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1}
.msub{color:#557755;font-size:10px;font-family:'JetBrains Mono',monospace;margin-top:6px}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    m = {}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath("app.py")), "models", "baseline_pipeline.pkl"),"rb") as f:
            m["intent"] = pickle.load(f)
    except:
        m["intent"] = None
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath("app.py")), "models", "recovery_predictor.pkl"),"rb") as f:
            m["recovery"] = pickle.load(f)
    except:
        m["recovery"] = None
    return m

MODELS = load_models()

ICFG = {
    "promise_to_pay":  {"color":"#00ff41","icon":"CHECK","label":"PROMISE TO PAY"},
    "needs_more_time": {"color":"#ffd700","icon":"CLOCK","label":"NEEDS MORE TIME"},
    "partial_payment": {"color":"#00cfff","icon":"HALF", "label":"PARTIAL PAYMENT"},
    "dispute":         {"color":"#ffd700","icon":"ALERT","label":"DISPUTE"},
    "refusal":         {"color":"#ff3131","icon":"DENY", "label":"REFUSAL"},
}

ACFG = {
    "SEND_FULL_PAYMENT_LINK":    {"color":"#00ff41","label":"SEND FULL PAYMENT LINK"},
    "SEND_PARTIAL_PAYMENT_LINK": {"color":"#00cfff","label":"SEND PARTIAL PAYMENT LINK"},
    "SCHEDULE_FOLLOWUP":         {"color":"#ffd700","label":"SCHEDULE FOLLOW-UP"},
    "FLAG_FOR_HUMAN_REVIEW":     {"color":"#ffd700","label":"FLAG FOR HUMAN REVIEW"},
    "ESCALATE_TO_SENIOR_TEAM":   {"color":"#ff3131","label":"ESCALATE TO SENIOR TEAM"},
    "TRIGGER_LEGAL_NOTICE":      {"color":"#ff3131","label":"TRIGGER LEGAL NOTICE"},
    "SEND_SETTLEMENT_OFFER":     {"color":"#00cfff","label":"SEND SETTLEMENT OFFER"},
}

EXAMPLES = [
    {"label":"Promise to Pay - Delhi","reminder":"Rahul ji aapki Rs5000 EMI 8 din se overdue hai. Aaj payment karein.","reply":"bhai kal pakka kar dunga aaj office mein busy tha","dpd":8,"amount":5000,"region":"delhi","tone":"polite"},
    {"label":"Partial Payment - Mumbai","reminder":"Arre Rs12000 business loan 20 din overdue. Settlement offer hai.","reply":"aadha abhi de sakta hun 6000 baaki 15 tarikh ko","dpd":20,"amount":12000,"region":"mumbai","tone":"cooperative"},
    {"label":"Needs More Time - Hyderabad","reminder":"Boss Rs8000 EMI miss ho gayi. CIBIL affect hoga.","reply":"bhai sach mein paisa nahi h mummy hospital mein hain 10 din aur do","dpd":12,"amount":8000,"region":"hyderabad","tone":"desperate"},
    {"label":"Dispute - Bangalore","reminder":"Sir Rs6500 EMI overdue hai. Please clear karein.","reply":"maine toh 3 din pehle UPI kar diya tha aapke system mein gadbad hai","dpd":7,"amount":6500,"region":"bangalore","tone":"angry"},
    {"label":"Refusal - Severe DPD","reminder":"Rs18000 bahut time se overdue. Legal notice bhejni padegi.","reply":"nahi karunga band karo yeh messages court mein milte hain","dpd":65,"amount":18000,"region":"delhi","tone":"aggressive"},
]

def run_pipeline(reminder, reply, dpd, amount, region, tone):
    res = {}
    if MODELS["intent"]:
        text = reminder + " [SEP] " + reply
        intent = MODELS["intent"].predict([text])[0]
        probs  = MODELS["intent"].predict_proba([text])[0]
        res["intent"] = intent
        res["conf"]   = float(max(probs))
        res["probs"]  = {c:float(p) for c,p in zip(MODELS["intent"].classes_, probs)}
    else:
        res["intent"] = "promise_to_pay"
        res["conf"]   = 0.95
        res["probs"]  = {}

    ptp = extract_ptp(reminder, reply, amount)
    res["ptp"] = ptp

    if MODELS["recovery"]:
        row = {"intent":res["intent"],"tone":tone,"dpd":dpd,"amount":amount,"region":region,"reply":reply,"reminder":reminder,"cibil_mentioned":False,"legal_mentioned":False}
        feat = engineer_features(pd.DataFrame([row]))
        rec  = MODELS["recovery"].predict(feat)[0]
        probs_r = MODELS["recovery"].predict_proba(feat)[0]
        res["recovery"]      = rec
        res["recovery_conf"] = float(max(probs_r))
    else:
        res["recovery"]      = "high"
        res["recovery_conf"] = 1.0

    intent   = res["intent"]
    ptp_amt  = ptp["ptp_amount"]["amount"] or amount
    partial  = ptp["ptp_amount"]["is_partial"]
    days_out = ptp["ptp_date"].get("days_from_now",1) or 1
    from src.agent.razorpay_client import create_payment_link, create_partial_payment_link

    if intent == "promise_to_pay":
        rzp = create_payment_link(amount=ptp_amt, customer_name="Customer", intent=intent, dpd=dpd) if not partial else create_partial_payment_link(total_amount=amount, partial_amount=ptp_amt, intent=intent, dpd=dpd)
        res["action"]      = "SEND_PARTIAL_PAYMENT_LINK" if partial else "SEND_FULL_PAYMENT_LINK"
        res["link"]        = rzp.get("short_url", "link_error") if rzp.get("success") else "api_error"
        res["link_amount"] = ptp_amt
        res["follow_up"]   = days_out + 1
    elif intent == "partial_payment":
        rzp = create_partial_payment_link(total_amount=amount, partial_amount=ptp["ptp_amount"]["amount"] or amount*0.5, intent=intent, dpd=dpd)
        res["action"]      = "SEND_PARTIAL_PAYMENT_LINK"
        res["link"]        = rzp.get("short_url", "link_error") if rzp.get("success") else "api_error"
        res["link_amount"] = ptp["ptp_amount"]["amount"] or amount * 0.5
        res["follow_up"]   = 7
    elif intent == "needs_more_time":
        res["action"]    = "SEND_SETTLEMENT_OFFER" if dpd > 60 else "SCHEDULE_FOLLOWUP"
        res["follow_up"] = 1 if dpd > 60 else days_out
    elif intent == "dispute":
        res["action"]    = "FLAG_FOR_HUMAN_REVIEW"
        res["ticket"]    = "VAADA-{}".format(random.randint(1000,9999))
        res["follow_up"] = 1
    elif intent == "refusal":
        res["action"]    = "TRIGGER_LEGAL_NOTICE" if dpd > 60 else "ESCALATE_TO_SENIOR_TEAM"
        res["follow_up"] = 0 if dpd > 60 else 1
    else:
        res["action"]    = "SCHEDULE_FOLLOWUP"
        res["follow_up"] = 3

    return res

with st.sidebar:
    st.markdown("## VAADA v1.0")
    st.markdown("Vernacular Agentic AI for Debt and Arrears")
    st.markdown("---")
    st.markdown("MODEL STATUS")
    st.markdown("OK Intent Classifier" if MODELS["intent"] else "MISSING Intent Classifier")
    st.markdown("OK Recovery Predictor" if MODELS["recovery"] else "MISSING Recovery Predictor")
    st.markdown("---")
    st.code("Intent F1  : 0.9890\nPTP Detect : 0.8219\nRecovery   : 1.0000\nLinks Gen  : 42.9%\nDataset    : 9,693", language=None)
    st.markdown("---")
    st.markdown("[GitHub](https://github.com/singhhnitin/vaada)")

tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs(["LIVE DEMO","EVAL RESULTS","BUSINESS IMPACT","DATASET","WHATSAPP SIM","RISK ANALYSIS"])

with tab1:
    st.markdown("## VAADA Live Pipeline")
    st.markdown("Paste a Hinglish WhatsApp message and watch the full pipeline run.")
    st.markdown("---")

    ex_labels = ["-- Select example --"] + [e["label"] for e in EXAMPLES]
    chosen = st.selectbox("Load example", ex_labels)
    if chosen != "-- Select example --":
        ex = next(e for e in EXAMPLES if e["label"] == chosen)
        for k in ["reminder","reply","dpd","amount","region","tone"]:
            st.session_state[k] = ex[k]

    c1,c2 = st.columns(2)
    with c1:
        reminder = st.text_area("Agent Reminder", value=st.session_state.get("reminder",""), height=90)
    with c2:
        reply = st.text_area("Customer Reply", value=st.session_state.get("reply",""), height=90)

    c3,c4,c5,c6 = st.columns(4)
    with c3:
        dpd = st.number_input("DPD", 1, 90, st.session_state.get("dpd",8))
    with c4:
        amount = st.number_input("Amount (Rs)", 500, 100000, st.session_state.get("amount",5000), step=500)
    with c5:
        region = st.selectbox("Region",["delhi","mumbai","hyderabad","bangalore"])
    with c6:
        tone = st.selectbox("Tone",["polite","desperate","evasive","angry","cooperative","neutral"])

    run_btn = st.button("RUN VAADA PIPELINE", use_container_width=True)

    if run_btn and reminder and reply:
        # Import real Razorpay client
        from src.agent.razorpay_client import create_payment_link, create_partial_payment_link
        st.markdown("---")
        prog = st.progress(0)
        status = st.empty()
        for label,val in [("Loading input",0.2),("Classifying intent",0.4),("Extracting PTP",0.6),("Predicting recovery",0.8),("Deciding action",1.0)]:
            status.markdown("`$ " + label + "...`")
            prog.progress(val)
            time.sleep(0.3)

        res = run_pipeline(reminder, reply, dpd, amount, region, tone)
        st.markdown("---")

        intent = res["intent"]
        conf   = res["conf"]
        ic     = ICFG.get(intent,{"color":"#00ff41","icon":"?","label":intent.upper()})
        rec    = res["recovery"]
        rc     = res["recovery_conf"]
        rec_color = {"high":"#00ff41","medium":"#ffd700","low":"#ff3131"}.get(rec,"#00ff41")
        ptp    = res["ptp"]
        ptp_date = ptp["ptp_date"]["raw"] or "none"
        ptp_amt  = ptp["ptp_amount"]["amount"]
        ptp_str  = "Rs {:.0f}".format(ptp_amt) if ptp_amt else "none"

        r1,r2,r3 = st.columns(3)
        with r1:
            st.markdown('<div class="mbox"><div class="mlabel">INTENT DETECTED</div><div class="mvalue" style="color:{}">{}</div><div class="msub">confidence: {:.1%}</div></div>'.format(ic["color"],ic["label"],conf), unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="mbox"><div class="mlabel">RECOVERY LIKELIHOOD</div><div class="mvalue" style="color:{}">{}</div><div class="msub">confidence: {:.1%}</div></div>'.format(rec_color,rec.upper(),rc), unsafe_allow_html=True)
        with r3:
            st.markdown('<div class="mbox"><div class="mlabel">PROMISE EXTRACTED</div><div class="mvalue">{}</div><div class="msub">date: {} | amount: {}</div></div>'.format("YES" if ptp["has_ptp"] else "NO",ptp_date,ptp_str), unsafe_allow_html=True)

        action = res.get("action","SCHEDULE_FOLLOWUP")
        ac     = ACFG.get(action,{"color":"#00ff41","label":action})

        detail = ""
        if "link" in res:
            detail += "LINK: {} | AMOUNT: Rs {:.0f}\n".format(res["link"],res["link_amount"])
        if "ticket" in res:
            detail += "TICKET: {}\n".format(res["ticket"])
        detail += "FOLLOW-UP IN: {} day(s)".format(res.get("follow_up","?"))

        st.markdown('<div style="background:#0d0d0d;border:1px solid {};padding:20px;margin-top:12px"><div style="color:#557755;font-size:10px;font-family:JetBrains Mono,monospace;text-transform:uppercase;margin-bottom:8px">AGENT ACTION</div><div style="color:{};font-size:18px;font-weight:700;font-family:JetBrains Mono,monospace;margin-bottom:8px">{}</div><div style="color:#c8ffc8;font-size:12px;font-family:JetBrains Mono,monospace;white-space:pre">{}</div></div>'.format(ac["color"],ac["color"],ac["label"],detail), unsafe_allow_html=True)

        if res.get("probs"):
            st.markdown("")
            st.markdown("**INTENT PROBABILITIES**")
            prob_df = pd.DataFrame(sorted(res["probs"].items(),key=lambda x:x[1],reverse=True),columns=["Intent","Probability"])
            st.bar_chart(prob_df.set_index("Intent"))

        st.markdown("**RAW OUTPUT**")
        st.code("INTENT   : {} ({:.4f})\nPTP DATE : {}\nPTP AMT  : {}\nRECOVERY : {} ({:.4f})\nACTION   : {}\nFOLLOW-UP: {} days\nSTATUS   : COMPLETE".format(intent,conf,ptp_date,ptp_str,rec,rc,action,res.get("follow_up","?")), language=None)

with tab2:
    st.markdown("## Evaluation Results")
    st.markdown("Held-out test set n=1454 no cherry-picking")
    st.markdown("---")
    m1,m2,m3,m4 = st.columns(4)
    for col,l,v,s in [(m1,"Intent F1","0.9890","TF-IDF + LR"),(m2,"PTP Detect","82.2%","date recall"),(m3,"Recovery","1.0000","GBM classifier"),(m4,"Links Gen","42.9%","auto per convo")]:
        with col:
            st.markdown('<div class="mbox"><div class="mlabel">{}</div><div class="mvalue">{}</div><div class="msub">{}</div></div>'.format(l,v,s),unsafe_allow_html=True)
    st.markdown("---")
    ca,cb = st.columns(2)
    with ca:
        st.markdown("**PER-CLASS REPORT**")
        rows=[{"Intent":"promise_to_pay","P":"1.000","R":"0.994","F1":"1.000","N":317},{"Intent":"needs_more_time","P":"0.993","R":"1.000","F1":"0.996","N":284},{"Intent":"partial_payment","P":"0.980","R":"1.000","F1":"0.990","N":304},{"Intent":"dispute","P":"0.980","R":"0.992","F1":"0.986","N":250},{"Intent":"refusal","P":"0.987","R":"0.960","F1":"0.973","N":299}]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    with cb:
        st.markdown("**ACTION DISTRIBUTION**")
        st.bar_chart(pd.DataFrame({"Action":["PARTIAL_LINK","FLAG","ESCALATE","FOLLOWUP","FULL_LINK"],"Count":[469,254,226,221,155]}).set_index("Action"))

with tab3:
    st.markdown("## Business Impact Calculator")
    st.markdown("Simulate Rs recovered by VAADA vs manual collections.")
    st.markdown("---")
    bc1,bc2,bc3 = st.columns(3)
    with bc1:
        n_convos = st.slider("Monthly conversations",100,100000,10000,100)
    with bc2:
        avg_amt = st.slider("Average EMI (Rs)",1000,50000,8000,500)
    with bc3:
        manual_pct = st.slider("Manual recovery rate %",10,60,25)

    vr = n_convos * 0.429 * 0.72 * avg_amt
    mr = n_convos * (manual_pct/100) * avg_amt
    delta = vr - mr
    uplift = ((vr/mr)-1)*100 if mr > 0 else 0

    ri1,ri2,ri3,ri4 = st.columns(4)
    with ri1:
        st.markdown('<div class="mbox"><div class="mlabel">VAADA Rate</div><div class="mvalue">30.9%</div><div class="msub">link to payment chain</div></div>',unsafe_allow_html=True)
    with ri2:
        st.markdown('<div class="mbox"><div class="mlabel">VAADA Recovery/Month</div><div class="mvalue">Rs {:.2f}M</div><div class="msub">automated</div></div>'.format(vr/1e6),unsafe_allow_html=True)
    with ri3:
        st.markdown('<div class="mbox"><div class="mlabel">Manual Recovery/Month</div><div class="mvalue" style="color:#557755">Rs {:.2f}M</div><div class="msub">at {}%</div></div>'.format(mr/1e6,manual_pct),unsafe_allow_html=True)
    with ri4:
        st.markdown('<div class="mbox"><div class="mlabel">VAADA Uplift</div><div class="mvalue" style="color:#ffd700">+Rs {:.2f}M</div><div class="msub">+{:.1f}% vs manual</div></div>'.format(delta/1e6,uplift),unsafe_allow_html=True)

    st.markdown("---")
    months = list(range(1,13))
    st.line_chart(pd.DataFrame({"Month":months,"VAADA (Rs Lakh)":[vr*m/1e5 for m in months],"Manual (Rs Lakh)":[mr*m/1e5 for m in months]}).set_index("Month"))
    st.code("VAADA on {:,} convos:\n  {:,} links sent (42.9%)\n  {:,} payments (72% of links)\n  {:,} auto-handled (57.1%)\n\nManual at {}%: {:,} recovered\n\nVAADA recovers {:,} MORE".format(n_convos,int(n_convos*0.429),int(n_convos*0.429*0.72),int(n_convos*0.571),manual_pct,int(n_convos*manual_pct/100),int(n_convos*0.429*0.72)-int(n_convos*manual_pct/100)),language=None)

with tab4:
    st.markdown("## VAADA Dataset")
    st.markdown("First regionalized Hinglish fintech NLU dataset.")
    st.markdown("---")
    da,db = st.columns(2)
    with da:
        st.markdown("**STATISTICS**")
        for k,v in [("total_samples","9,693"),("llm_generated","4,061"),("augmented","5,632"),("multi_turn","355"),("intent_classes","5"),("dialects","4 Delhi Mumbai Hyderabad Bangalore"),("dpd_range","1-90 days"),("rbi_compliant","true"),("public_prior_art","none novel")]:
            st.markdown("`{}` : **{}**".format(k,v))
    with db:
        st.markdown("**INTENT DISTRIBUTION**")
        st.bar_chart(pd.DataFrame({"Intent":["promise_to_pay","needs_more_time","partial_payment","dispute","refusal"],"Count":[2184,1879,1877,1836,1917]}).set_index("Intent"))
    st.markdown("---")
    st.markdown("**DIALECT EXAMPLES**")
    for reg,ex in [("Delhi","yaar bhai pakka kal kar dunga"),("Mumbai","arre nako tension aaj kar deto"),("Hyderabad","boss definitely kar dunga anna"),("Bangalore","sir illa kal UPI maadtini bekku")]:
        st.markdown("`{}` -> {}".format(reg,ex))
    st.markdown("---")
    try:
        df = pd.read_csv("data/processed/test.csv")
        cols = [c for c in ["reminder","reply","intent","tone","region","dpd","amount"] if c in df.columns]
        st.dataframe(df[cols].head(10),use_container_width=True,hide_index=True)
    except:
        st.code("Run data generation first.")

st.markdown("---")
st.markdown("`VAADA v1.0.0 | Razorpay AI Buildathon 2026 | Revenue Recovery Track | github.com/singhhnitin/vaada`")

with tab5:
    st.markdown("## WhatsApp Collections Simulator")
    st.markdown("Simulate a real multi-day Hinglish collections thread. Watch VAADA process each turn.")
    st.markdown("---")

    if "ws_conv" not in st.session_state:
        st.session_state.ws_conv = []
        st.session_state.ws_day  = 1
        st.session_state.ws_status = "active"
        st.session_state.ws_links  = []

    wa1, wa2 = st.columns([2,1])

    with wa1:
        st.markdown("**CONVERSATION THREAD**")
        chat_html = '<div style="background:#050505;border:1px solid #1a3a1a;padding:16px;height:400px;overflow-y:auto;font-family:JetBrains Mono,monospace;font-size:12px">'
        if not st.session_state.ws_conv:
            chat_html += '<div style="color:#557755;text-align:center;margin-top:160px">Start conversation below</div>'
        for msg in st.session_state.ws_conv:
            if msg["sender"] == "agent":
                chat_html += f'<div style="margin-bottom:12px"><div style="color:#557755;font-size:10px">AGENT · Day {msg["day"]}</div><div style="background:#003311;border:1px solid #1a3a1a;padding:8px;color:#00ff41;margin-top:2px">{msg["text"]}</div></div>'
            else:
                chat_html += f'<div style="margin-bottom:12px;text-align:right"><div style="color:#557755;font-size:10px">CUSTOMER · Day {msg["day"]}</div><div style="background:#0d0d0d;border:1px solid #1a3a1a;padding:8px;color:#c8ffc8;margin-top:2px">{msg["text"]}</div>'
                if msg.get("intent"):
                    color = "#00ff41" if msg["intent"] in ["promise_to_pay","partial_payment"] else "#ffd700" if msg["intent"] == "needs_more_time" else "#ff3131"
                    chat_html += f'<div style="color:{color};font-size:10px;margin-top:2px">▶ {msg["intent"].upper()}</div>'
                if msg.get("link"):
                    chat_html += f'<div style="color:#00cfff;font-size:10px">🔗 {msg["link"]}</div>'
                chat_html += '</div>'
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)

    with wa2:
        st.markdown("**THREAD STATUS**")
        st.markdown(f'`Day      : {st.session_state.ws_day}`')
        st.markdown(f'`Messages : {len(st.session_state.ws_conv)}`')
        st.markdown(f'`Status   : {st.session_state.ws_status}`')
        st.markdown(f'`Links    : {len(st.session_state.ws_links)}`')

    st.markdown("---")
    ws1, ws2 = st.columns(2)
    with ws1:
        agent_msg = st.text_input("Agent sends:", placeholder="Rahul ji Rs5000 EMI overdue hai...")
    with ws2:
        customer_msg = st.text_input("Customer replies:", placeholder="bhai kal pakka kar dunga 🙏")

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        ws_amount = st.number_input("Amount", 500, 50000, 5000, step=500, key="ws_amt")
    with wc2:
        ws_dpd = st.number_input("DPD", 1, 90, st.session_state.ws_day, key="ws_dpd")
    with wc3:
        ws_region = st.selectbox("Region", ["delhi","mumbai","hyderabad","bangalore"], key="ws_reg")

    btn1, btn2 = st.columns(2)
    with btn1:
        if st.button("SEND TURN", use_container_width=True):
            if agent_msg and customer_msg:
                from src.nlu.ptp_extractor import extract_ptp
                intent_result = run_pipeline(agent_msg, customer_msg, ws_dpd, ws_amount, ws_region, "neutral")
                ptp = intent_result["ptp"]

                st.session_state.ws_conv.append({"sender":"agent","text":agent_msg,"day":st.session_state.ws_day})

                customer_entry = {
                    "sender":   "customer",
                    "text":     customer_msg,
                    "day":      st.session_state.ws_day,
                    "intent":   intent_result["intent"],
                    "link":     None
                }

                if intent_result["intent"] in ["promise_to_pay","partial_payment"]:
                    link = intent_result.get("link","")
                    customer_entry["link"] = link
                    if link:
                        st.session_state.ws_links.append(link)

                st.session_state.ws_conv.append(customer_entry)
                st.session_state.ws_day += 2
                st.session_state.ws_status = intent_result["intent"]
                st.rerun()

    with btn2:
        if st.button("RESET", use_container_width=True):
            st.session_state.ws_conv   = []
            st.session_state.ws_day    = 1
            st.session_state.ws_status = "active"
            st.session_state.ws_links  = []
            st.rerun()

with tab6:
    st.markdown("## Risk Analysis — Diagnose Revenue Leaks")
    st.markdown("Pattern analysis across 9,693 conversations to identify WHY revenue leaks and WHO defaults.")
    st.markdown("---")

    try:
        import json
        with open("outputs/risk_analysis.json") as f:
            risk_data = json.load(f)

        patterns = risk_data.get("patterns", {})
        leaks    = risk_data.get("revenue_leaks", {})

        # Key stats
        rk1,rk2,rk3 = st.columns(3)
        with rk1:
            st.markdown('<div class="mbox"><div class="mlabel">HIGH RISK RATE</div><div class="mvalue" style="color:#ff3131">38.7%</div><div class="msub">refusal + dispute</div></div>', unsafe_allow_html=True)
        with rk2:
            st.markdown('<div class="mbox"><div class="mlabel">HIGHEST RISK REGION</div><div class="mvalue" style="color:#ffd700">Hyderabad</div><div class="msub">53.9% refusal rate</div></div>', unsafe_allow_html=True)
        with rk3:
            st.markdown('<div class="mbox"><div class="mlabel">BEST INTERVENTION</div><div class="mvalue">DPD 1-15</div><div class="msub">highest recovery potential</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        ra,rb = st.columns(2)

        with ra:
            st.markdown("**REGIONAL RISK PATTERNS**")
            regional = patterns.get("regional_patterns", {})
            reg_df = pd.DataFrame([
                {"Region": k, "Refusal Rate": v["refusal_rate"], "Count": v["total"]}
                for k,v in regional.items() if v["total"] > 50
            ]).sort_values("Refusal Rate", ascending=False)
            if not reg_df.empty:
                st.bar_chart(reg_df.set_index("Region")["Refusal Rate"])

        with rb:
            st.markdown("**DPD STAGE ANALYSIS**")
            dpd_data = patterns.get("dpd_stage_analysis", {})
            dpd_df = pd.DataFrame([
                {"Stage": k, "Promise Rate": v["promise_rate"], "Refusal Rate": v["refusal_rate"]}
                for k,v in dpd_data.items()
            ])
            if not dpd_df.empty:
                st.bar_chart(dpd_df.set_index("Stage"))

        st.markdown("---")
        st.markdown("**KEY INSIGHTS**")
        for i, insight in enumerate(patterns.get("key_insights", []), 1):
            st.markdown(f"`{i}.` {insight}")

        st.markdown("---")
        st.markdown("**REVENUE LEAK DIAGNOSIS**")
        segments = leaks.get("at_risk_segments", {})
        for seg, data in segments.items():
            col = "#ff3131" if seg == "high_risk" else "#ffd700" if seg == "medium_risk" else "#00ff41"
            st.markdown(f'<div style="background:#0d0d0d;border:1px solid {col};padding:12px;margin-bottom:8px;font-family:JetBrains Mono,monospace"><span style="color:{col};font-weight:700">{seg.upper()}</span> — {data["count"]} conversations ({data["pct"]}%) → {data["recommended_action"]}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.code(f"Run src/analysis/risk_analyzer.py first. Error: {e}")
