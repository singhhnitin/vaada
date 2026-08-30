# ── TAB 1: LIVE DEMO ─────────────────────────────────────────
with tab1:
    st.markdown("## VAADA Live Pipeline")
    st.markdown("Paste a Hinglish WhatsApp reply. VAADA classifies intent, extracts the promise, and generates a real Razorpay link.")
    st.markdown("---")

    # Reordered: promise_to_pay first — it's the strongest visual (real payment link)
    EXAMPLES_ORDERED = [
        EXAMPLES[0],  # Promise to Pay - Delhi
        EXAMPLES[1],  # Partial Payment - Mumbai
        EXAMPLES[2],  # Needs More Time - Hyderabad
        EXAMPLES[3],  # Dispute - Bangalore
        EXAMPLES[4],  # Refusal - Severe DPD
    ]

    ex_labels = [e["label"] for e in EXAMPLES_ORDERED]
    default_idx = 0  # promise_to_pay loads by default
    chosen = st.selectbox("Load example", ex_labels, index=default_idx)
    ex = next(e for e in EXAMPLES_ORDERED if e["label"] == chosen)
    for k in ["reminder", "reply", "dpd", "amount", "region", "tone"]:
        if k not in st.session_state or st.session_state.get("_last_example") != chosen:
            st.session_state[k] = ex[k]
    st.session_state["_last_example"] = chosen

    reply = st.text_area(
        "Customer's Hinglish message",
        value=st.session_state.get("reply", ""),
        height=100,
        placeholder="bhai kal pakka kar dunga aaj busy tha 🙏"
    )

    with st.expander("⚙ ADVANCED — reminder text, DPD, amount, region, tone"):
        reminder = st.text_area("Agent Reminder", value=st.session_state.get("reminder", ""), height=70)
        c3, c4, c5, c6 = st.columns(4)
        with c3:
            dpd = st.number_input("DPD", 1, 90, st.session_state.get("dpd", 8))
        with c4:
            amount = st.number_input("Amount (Rs)", 500, 100000, st.session_state.get("amount", 5000), step=500)
        with c5:
            region = st.selectbox("Region", ["delhi", "mumbai", "hyderabad", "bangalore"],
                                   index=["delhi", "mumbai", "hyderabad", "bangalore"].index(st.session_state.get("region", "delhi")))
        with c6:
            tone = st.selectbox("Tone", ["polite", "desperate", "evasive", "angry", "cooperative", "neutral"],
                                 index=["polite", "desperate", "evasive", "angry", "cooperative", "neutral"].index(st.session_state.get("tone", "polite")) if st.session_state.get("tone") in ["polite","desperate","evasive","angry","cooperative","neutral"] else 0)

    run_btn = st.button("▶ RUN VAADA PIPELINE", use_container_width=True, type="primary")

    if run_btn and reply:
        st.markdown("---")
        prog = st.progress(0)
        status = st.empty()
        for label, val in [("Loading input", 0.2), ("Classifying intent", 0.4), ("Extracting PTP", 0.6),
                            ("Predicting recovery", 0.8), ("Deciding action", 1.0)]:
            status.markdown("`$ " + label + "...`")
            prog.progress(val)
            time.sleep(0.25)
        status.empty()
        prog.empty()

        res = run_pipeline(reminder, reply, dpd, amount, region, tone)
        st.markdown("---")
        st.markdown("### ✓ PIPELINE OUTPUT")

        intent = res["intent"]
        conf   = res["conf"]
        ic     = ICFG.get(intent, {"color": "#00ff41", "label": intent.upper()})
        rec    = res["recovery"]
        rc     = res["recovery_conf"]
        rec_color = {"high": "#00ff41", "medium": "#ffd700", "low": "#ff3131"}.get(rec, "#00ff41")
        ptp    = res["ptp"]
        ptp_date = ptp["ptp_date"]["raw"] or "none"
        ptp_amt  = ptp["ptp_amount"]["amount"]
        ptp_str  = "Rs {:.0f}".format(ptp_amt) if ptp_amt else "none"

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown('<div class="mbox"><div class="mlabel">INTENT DETECTED</div><div class="mvalue" style="color:{}">{}</div><div class="msub">confidence: {:.1%}</div></div>'.format(ic["color"], ic["label"], conf), unsafe_allow_html=True)
        with r2:
            st.markdown('<div class="mbox"><div class="mlabel">RECOVERY LIKELIHOOD</div><div class="mvalue" style="color:{}">{}</div><div class="msub">confidence: {:.1%}</div></div>'.format(rec_color, rec.upper(), rc), unsafe_allow_html=True)
        with r3:
            st.markdown('<div class="mbox"><div class="mlabel">PROMISE EXTRACTED</div><div class="mvalue">{}</div><div class="msub">date: {} | amount: {}</div></div>'.format("YES" if ptp["has_ptp"] else "NO", ptp_date, ptp_str), unsafe_allow_html=True)

        action = res.get("action", "SCHEDULE_FOLLOWUP")
        ac     = ACFG.get(action, {"color": "#00ff41", "label": action})

        # ── Loud, unmissable action panel ──
        if "link" in res:
            st.markdown(
                '<div style="background:#001a0a;border:2px solid #00ff41;padding:24px;margin-top:16px">'
                '<div style="color:#557755;font-size:10px;font-family:JetBrains Mono,monospace;text-transform:uppercase;margin-bottom:10px">RAZORPAY PAYMENT LINK GENERATED</div>'
                '<div style="color:#00ff41;font-size:22px;font-weight:700;font-family:JetBrains Mono,monospace;word-break:break-all">{}</div>'
                '<div style="color:#c8ffc8;font-size:13px;font-family:JetBrains Mono,monospace;margin-top:10px">Amount: Rs {:.0f} | Expires in 24h | Follow-up in {} day(s)</div>'
                '</div>'.format(res["link"], res["link_amount"], res.get("follow_up", "?")),
                unsafe_allow_html=True
            )
        else:
            detail = ""
            if "ticket" in res:
                detail += "TICKET: {}\n".format(res["ticket"])
            detail += "FOLLOW-UP IN: {} day(s)".format(res.get("follow_up", "?"))
            st.markdown(
                '<div style="background:#0d0d0d;border:1px solid {};padding:20px;margin-top:12px">'
                '<div style="color:#557755;font-size:10px;font-family:JetBrains Mono,monospace;text-transform:uppercase;margin-bottom:8px">AGENT ACTION</div>'
                '<div style="color:{};font-size:18px;font-weight:700;font-family:JetBrains Mono,monospace;margin-bottom:8px">{}</div>'
                '<div style="color:#c8ffc8;font-size:12px;font-family:JetBrains Mono,monospace;white-space:pre">{}</div>'
                '</div>'.format(ac["color"], ac["color"], ac["label"], detail),
                unsafe_allow_html=True
            )

        if res.get("probs"):
            st.markdown("")
            st.markdown("**INTENT PROBABILITIES**")
            prob_df = pd.DataFrame(sorted(res["probs"].items(), key=lambda x: x[1], reverse=True), columns=["Intent", "Probability"])
            st.bar_chart(prob_df.set_index("Intent"))

        with st.expander("RAW OUTPUT"):
            st.code("INTENT   : {} ({:.4f})\nPTP DATE : {}\nPTP AMT  : {}\nRECOVERY : {} ({:.4f})\nACTION   : {}\nFOLLOW-UP: {} days\nSTATUS   : COMPLETE".format(intent, conf, ptp_date, ptp_str, rec, rc, action, res.get("follow_up", "?")), language=None)
    elif run_btn and not reply:
        st.warning("Paste a customer message first.")
