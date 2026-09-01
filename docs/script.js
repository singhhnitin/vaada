const V = { vermilion:'#E4552B', mint:'#5FD6A4', rose:'#E8607E', ink:'#191621', mid:'rgba(25,22,33,0.62)' };

const API_BASE = 'https://technology-measuring-pets-daniel.trycloudflare.com';

const $ = s => document.querySelector(s);

const REGION_EXAMPLES = {
  delhi: { reminder: "Rahul ji aapki Rs5000 EMI 8 din se overdue hai. Aaj payment karein.", reply: "bhai kal pakka kar dunga aaj office mein busy tha" },
  mumbai: { reminder: "Arre Rs12000 business loan 20 din overdue. Settlement offer hai.", reply: "aadha abhi de sakta hun 6000 baaki 15 tarikh ko" },
  hyderabad: { reminder: "Boss Rs8000 EMI miss ho gayi. CIBIL affect hoga.", reply: "bhai sach mein paisa nahi h mummy hospital mein hain 10 din aur do" },
  bangalore: { reminder: "Sir Rs6500 EMI overdue hai. Please clear karein.", reply: "maine toh 3 din pehle UPI kar diya tha aapke system mein gadbad hai" }
};

function loadRegionExample(region) {
  const ex = REGION_EXAMPLES[region] || REGION_EXAMPLES.delhi;
  document.querySelector("#in-reminder").value = ex.reminder;
  document.querySelector("#in-reply").value = ex.reply;
}

// ── Check API health on load ────────────────────────────────
// Load default example on page load, and update when region changes
loadRegionExample('delhi');
document.querySelector('#in-region').addEventListener('change', (e) => loadRegionExample(e.target.value));

fetch(API_BASE + '/api/health')
  .then(r => r.json())
  .then(d => {
    $('#api-status').textContent = d.status === 'ok' ? 'API: LIVE' : 'API: ERROR';
    $('#api-status').style.color = d.status === 'ok' ? V.mint : V.rose;
  })
  .catch(() => {
    $('#api-status').textContent = 'API: OFFLINE';
    $('#api-status').style.color = V.rose;
  });

// ── Intent color map ─────────────────────────────────────────
const INTENT_COLOR = {
  promise_to_pay: V.vermilion,
  partial_payment: V.mint,
  needs_more_time: 'rgba(242,239,233,.7)',
  dispute: '#E8B84A',
  refusal: V.rose
};

const ACTION_LABEL = {
  SEND_FULL_PAYMENT_LINK: 'FULL PAYMENT LINK ISSUED',
  SEND_PARTIAL_PAYMENT_LINK: 'PARTIAL PAYMENT LINK ISSUED',
  SCHEDULE_FOLLOWUP: 'FOLLOW-UP SCHEDULED',
  SEND_SETTLEMENT_OFFER: 'SETTLEMENT OFFER SENT',
  FLAG_FOR_HUMAN_REVIEW: 'FLAGGED FOR HUMAN REVIEW',
  ESCALATE_TO_SENIOR_TEAM: 'ESCALATED TO SENIOR TEAM',
  TRIGGER_LEGAL_NOTICE: 'LEGAL NOTICE TRIGGERED'
};

// ── Build the "tape" lines from a real API response ─────────
function buildTapeFromResult(reminder, reply, region, tone, dpd, amount, result) {
  const L = (k, v, o = {}) => Object.assign({ k, v, size: 12.5, color: V.ink, weight: 400, track: .02, rule: false }, o);
  const now = new Date();
  const timeStr = now.toISOString().slice(0, 16).replace('T', ' ');

  const lines = [
    L('READ AT', timeStr, { color: V.mid, size: 11 }),
    L('CHANNEL', 'WHATSAPP · TEXT', { color: V.mid, size: 11 }),
    L('DIALECT', 'HINGLISH / ' + region.toUpperCase(), { color: V.mid, size: 11, rule: true }),
    L('INTENT', result.intent.replace(/_/g, ' ').toUpperCase(), { size: 16, weight: 500, track: .06 }),
    L('CLASSIFIER', 'TF-IDF + LR · ' + result.confidence.toFixed(2), { color: V.mid, size: 11 }),
    L('TONE ROUTED', tone.toUpperCase() + ' · DPD ' + dpd, { color: V.mid, size: 11, rule: true }),
  ];

  if (result.ptp_date) {
    lines.push(L('PROMISED DATE', result.ptp_date, { size: 15, weight: 500 }));
  }
  if (result.ptp_amount) {
    lines.push(L('PROMISED AMT', '₹' + result.ptp_amount.toLocaleString(), { size: 20, weight: 500, rule: true }));
  }
  lines.push(L('EMI DUE', '₹' + parseFloat(amount).toLocaleString(), { color: V.mid, size: 11 }));
  lines.push(L('RECOVERY SCORE', result.recovery.toUpperCase() + ' · ' + result.recovery_confidence.toFixed(2), { size: 12.5, weight: 500, rule: true }));
  lines.push(L('ACTION', ACTION_LABEL[result.action] || result.action, { size: 12.5, weight: 500, color: '#C2431F' }));

  return lines;
}

// ── Render live result into the existing DOM structure ──────
function renderLiveResult(reminder, reply, region, tone, dpd, amount, result) {
  const tintColor = INTENT_COLOR[result.intent] || V.vermilion;

  // Thread (left column)
  $('#thread').innerHTML = `
    <div class="msg on">
      <div class="msg-time">JUST NOW</div>
      <div>
        <div class="msg-who" style="color:rgba(242,239,233,0.4)">VAADA / OUTBOUND</div>
        <div class="msg-text" style="border-left-color:rgba(242,239,233,0.4)">${reminder || '(no reminder text entered)'}</div>
        <div class="msg-meta">SENT</div>
      </div>
    </div>
    <div class="msg on">
      <div class="msg-time">JUST NOW</div>
      <div>
        <div class="msg-who" style="color:${tintColor}">CUSTOMER</div>
        <div class="msg-text" style="border-left-color:${tintColor}">${reply}</div>
        <div class="msg-meta">CLASSIFIED · ${result.intent.replace(/_/g, ' ').toUpperCase()} · LIVE MODEL</div>
      </div>
    </div>`;

  // Tape (center)
  const tapeLines = buildTapeFromResult(reminder, reply, region, tone, dpd, amount, result);
  const tape = $('#tape');
  tape.innerHTML = '';
  tapeLines.forEach(l => {
    const row = document.createElement('div');
    row.className = 'tline' + (l.rule ? ' rule' : '');
    row.style.cssText = `font-size:${l.size}px;color:${l.color};font-weight:${l.weight};letter-spacing:${l.track}em`;
    row.innerHTML = `<span>${l.k}</span><span>${l.v}</span>`;
    tape.appendChild(row);
  });
  $('#printed').textContent = `PRINTED ${tapeLines.length} / ${tapeLines.length} LINES`;
  $('#caret').hidden = true;
  $('#feeding').classList.remove('on');

  // Stub (payment link) — only show if a real link was generated
  const stub = $('#stub');
  if (result.link) {
    $('#link-out').textContent = result.link.replace('https://', '');
    $('#link-out').href = result.link;
    stub.querySelector('.stub-fine').textContent =
      `AMOUNT ₹${(result.link_amount || amount).toLocaleString()} · EXPIRES 24H · SENT TO THE SAME THREAD`;
    stub.classList.add('on');
  } else {
    stub.classList.remove('on');
  }

  // Notes (right column) — analysis cards
  const notesHtml = [];
  notesHtml.push(`
    <div class="ev on">
      <div class="ev-head">
        <i style="background:${tintColor}"></i>
        <span class="ev-label" style="color:${tintColor}">INTENT</span>
        <span class="ev-conf">${result.confidence.toFixed(2)}</span>
      </div>
      <div class="ev-value">${result.intent.replace(/_/g, ' ')}</div>
      <div class="ev-basis">Classified live by the TF-IDF + Logistic Regression model, trained on 9,693 Hinglish collections examples.</div>
    </div>`);

  if (result.ptp_date || result.ptp_amount) {
    notesHtml.push(`
      <div class="ev on">
        <div class="ev-head">
          <i style="background:${V.vermilion}"></i>
          <span class="ev-label" style="color:${V.vermilion}">PROMISE EXTRACTED</span>
          <span class="ev-conf">CRF MODEL</span>
        </div>
        <div class="ev-value">${result.ptp_date || 'no date'} ${result.ptp_amount ? '· ₹' + result.ptp_amount.toLocaleString() : ''}</div>
        <div class="ev-basis">Extracted live from the raw reply text using a trained Conditional Random Field model.</div>
      </div>`);
  }

  notesHtml.push(`
    <div class="ev on">
      <div class="ev-head">
        <i style="background:${V.mint}"></i>
        <span class="ev-label" style="color:${V.mint}">ACTION TAKEN</span>
        <span class="ev-conf">LIVE</span>
      </div>
      <div class="ev-value">${ACTION_LABEL[result.action] || result.action}</div>
      <div class="ev-basis">${result.link ? 'A real Razorpay Test-mode payment link was generated for this promise.' : 'Routed per the decision policy for this intent — no payment link needed for this case.'}</div>
    </div>`);

  $('#notes').innerHTML = notesHtml.join('');
  $('#guard').classList.add('on');
}

// ── Wire the run button to the real API ──────────────────────
async function runLive() {
  const reply = $('#in-reply').value.trim();
  const reminder = $('#in-reminder').value.trim();
  const region = $('#in-region').value;
  const tone = $('#in-tone').value;
  const dpd = parseInt($('#in-dpd').value || '8', 10);
  const amount = parseFloat($('#in-amount').value || '5000');

  if (!reply) {
    alert('Type a customer reply first.');
    return;
  }

  const btn = $('#run');
  const originalText = btn.textContent;
  btn.textContent = '▷ RUNNING...';
  btn.disabled = true;

  try {
    const res = await fetch(API_BASE + '/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reminder, reply, dpd, amount, region, tone })
    });
    const result = await res.json();
    if (result.error) {
      alert('Error: ' + result.error);
    } else {
      renderLiveResult(reminder, reply, region, tone, dpd, amount, result);
    }
  } catch (err) {
    alert('Could not reach the API. It may be offline — check the API status indicator.');
  } finally {
    btn.textContent = originalText;
    btn.disabled = false;
  }
}

$('#run').addEventListener('click', runLive);

// ── Register table stays as illustrative static data ─────────
const REGISTER = [
  ['0114','R. KUMAR','"6 September tak 5000 pakka"','PROMISE TO PAY','06','₹5,000','0.96','RECOVERABLE',V.mint],
  ['0113','S. IYER','"salary aane par pura kar dungi"','PROMISE TO PAY','11','₹12,400','0.71','MEDIUM',V.vermilion],
  ['0111','M. ANSARI','"aaj hi pay kar diya, receipt dekh lo"','DISPUTE','02','₹3,150','0.99','HUMAN REVIEW',V.mint],
  ['0109','P. NAIK','"abhi paisa nahi hai, next month dekhenge"','PARTIAL / STALL','34','₹8,900','0.88','HIGH',V.rose],
  ['0106','D. CHAUHAN','"half 4th ko, half 20th ko"','PARTIAL PAYMENT','09','₹2,500','0.93','RECOVERABLE',V.mint],
  ['0104','A. BOSE','"link kaam nahi kar raha sir"','PROMISE TO PAY','07','₹6,720','0.90','LINK REISSUED',V.mint],
  ['0102','K. RATHOD','"main nahi doonga, court jao"','REFUSAL','61','₹1,480','0.95','LEGAL ROUTE',V.rose]
];
const BARS = [2,1,4,1,2,5,1,3,1,2,4,1,1,3,2,1,5,1,2,3,1,4];

(function buildStatic(){
  $('#register').innerHTML = REGISTER.map(r=>`
    <tr>
      <td class="folio">${r[0]}</td><td>${r[1]}</td><td class="quote">${r[2]}</td>
      <td class="intent">${r[3]}</td><td>${r[4]}</td><td>${r[5]}</td>
      <td style="color:rgba(242,239,233,.55)">${r[6]}</td>
      <td><span class="risk" style="border-bottom-color:${r[8]}">${r[7]}</span></td>
    </tr>`).join('');
  $('#barcode').innerHTML = BARS.map(w=>`<i style="flex:0 0 ${w}px"></i>`).join('');
})();
