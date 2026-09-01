const V = { vermilion:'#E4552B', mint:'#5FD6A4', rose:'#E8607E', ink:'#191621', mid:'rgba(25,22,33,0.62)' };

const THREAD = [
  { time:'28 AUG 10:02', who:'VAADA / OUTBOUND', tint:'rgba(242,239,233,0.4)',
    text:'Namaste Rakesh ji. Aapki August EMI ₹4,820 ki due date nikal gayi hai. Kab tak kar payenge?',
    meta:'SOFT TONE · DPD 1' },
  { time:'30 AUG 21:40', who:'R. KUMAR', tint:'rgba(242,239,233,0.55)',
    text:'thoda time chahiye sir, dekhta hoon',
    meta:'CLASSIFIED · SILENCE / STALL · NO PROMISE LOGGED' },
  { time:'02 SEP 10:14', who:'R. KUMAR', tint:V.vermilion,
    text:'sir salary 5 tarikh ko aa jaati hai, 6 September tak 5000 pakka kar dunga. link bhej dijiye',
    meta:'CLASSIFIED · PROMISE TO PAY · TAPE PRINTING' }
];

const L = (k,v,o={}) => Object.assign({ k,v,size:12.5,color:V.ink,weight:400,track:.02,rule:false }, o);
const TAPE = [
  L('READ AT','02 SEP 10:14 IST',{color:V.mid,size:11}),
  L('CHANNEL','WHATSAPP · TEXT',{color:V.mid,size:11}),
  L('DIALECT','HINGLISH / DELHI',{color:V.mid,size:11,rule:true}),
  L('INTENT','PROMISE TO PAY',{size:16,weight:500,track:.06}),
  L('CLASSIFIER','TF-IDF + LR · 0.96',{color:V.mid,size:11}),
  L('TONE ROUTED','SOFT · DPD 6',{color:V.mid,size:11,rule:true}),
  L('PROMISED DATE','06 SEP 2026',{size:15,weight:500}),
  L('PROMISED AMT','₹5,000.00',{size:20,weight:500,rule:true}),
  L('EMI DUE','₹4,820.00',{color:V.mid,size:11}),
  L('AFTER PAYMENT','₹0.00 · EMI CLOSED',{color:V.mid,size:11,rule:true}),
  L('BASIS','“6 September tak”',{color:V.mid,size:11}),
  L('BASIS','“5000 pakka”',{color:V.mid,size:11}),
  L('BASIS','“link bhej dijiye”',{color:V.mid,size:11,rule:true}),
  L('RISK SCORE','RECOVERABLE · 0.81',{size:12.5,weight:500}),
  L('DISPOSITION','PTP LOGGED · F/U 06 SEP',{size:12.5,weight:500,color:'#C2431F'})
];

const NOTES = [
  { label:'INTENT', value:'Promise to pay', conf:'0.96', tint:V.vermilion, at:6,
    basis:'One of five classes — promise, refusal, dispute, partial, silence. “Dekh lunga” would have landed as silence, not a promise.' },
  { label:'DATE LIFTED', value:'06 Sep 2026', conf:'0.98', tint:V.vermilion, at:10,
    basis:'“6 September tak” resolved against the stated salary date (5th) and this month’s calendar, then held as the follow-up trigger.' },
  { label:'AMOUNT LIFTED', value:'₹5,000.00', conf:'0.94', tint:V.vermilion, at:12,
    basis:'Bare “5000” read as rupees, above the ₹4,820 due, so the entry is marked as closing the EMI rather than a part payment.' },
  { label:'ACTION TAKEN', value:'Razorpay link', conf:'ISSUED', tint:V.mint, at:TAPE.length+4,
    basis:'Explicit consent to be sent a link, so the tape prints one for the promised figure — partial payment on, expiring with the promise.' }
];

const REGISTER = [
  ['0114','R. KUMAR','“6 September tak 5000 pakka”','PROMISE TO PAY','06','₹5,000','0.96','RECOVERABLE',V.mint],
  ['0113','S. IYER','“salary aane par pura kar dungi”','PROMISE TO PAY','11','₹12,400','0.71','MEDIUM',V.vermilion],
  ['0111','M. ANSARI','“aaj hi pay kar diya, receipt dekh lo”','DISPUTE','02','₹3,150','0.99','HUMAN REVIEW',V.mint],
  ['0109','P. NAIK','“abhi paisa nahi hai, next month dekhenge”','PARTIAL / STALL','34','₹8,900','0.88','HIGH',V.rose],
  ['0106','D. CHAUHAN','“half 4th ko, half 20th ko”','PARTIAL PAYMENT','09','₹2,500','0.93','RECOVERABLE',V.mint],
  ['0104','A. BOSE','“link kaam nahi kar raha sir”','PROMISE TO PAY','07','₹6,720','0.90','LINK REISSUED',V.mint],
  ['0102','K. RATHOD','“main nahi doonga, court jao”','REFUSAL','61','₹1,480','0.95','LEGAL ROUTE',V.rose]
];

const BARS = [2,1,4,1,2,5,1,3,1,2,4,1,1,3,2,1,5,1,2,3,1,4];
const $ = s => document.querySelector(s);

/* static render */
(function build(){
  $('#thread').innerHTML = THREAD.map((m,i)=>`
    <div class="msg" data-i="${i}">
      <div class="msg-time">${m.time}</div>
      <div>
        <div class="msg-who" style="color:${m.tint}">${m.who}</div>
        <div class="msg-text" style="border-left-color:${m.tint}">${m.text}</div>
        <div class="msg-meta">${m.meta}</div>
      </div>
    </div>`).join('');

  $('#notes').innerHTML = NOTES.map((n,i)=>`
    <div class="ev" data-i="${i}">
      <div class="ev-head">
        <i style="background:${n.tint}"></i>
        <span class="ev-label" style="color:${n.tint}">${n.label}</span>
        <span class="ev-conf">${n.conf}</span>
      </div>
      <div class="ev-value">${n.value}</div>
      <div class="ev-basis">${n.basis}</div>
    </div>`).join('');

  $('#register').innerHTML = REGISTER.map(r=>`
    <tr>
      <td class="folio">${r[0]}</td><td>${r[1]}</td><td class="quote">${r[2]}</td>
      <td class="intent">${r[3]}</td><td>${r[4]}</td><td>${r[5]}</td>
      <td style="color:rgba(242,239,233,.55)">${r[6]}</td>
      <td><span class="risk" style="border-bottom-color:${r[8]}">${r[7]}</span></td>
    </tr>`).join('');

  $('#barcode').innerHTML = BARS.map(w=>`<i style="flex:0 0 ${w}px"></i>`).join('');
  $('#printed').textContent = `PRINTED 0 / ${TAPE.length} LINES`;
})();

/* timeline: 1-3 inbound rows, then one tape line per tick, then the stub */
let timers = [], step = 0;

function paint(){
  document.querySelectorAll('.msg').forEach(el=>{
    el.classList.toggle('on', step >= Number(el.dataset.i) + 1);
  });

  const printed = Math.max(0, Math.min(TAPE.length, step - 3));
  const done = step > TAPE.length + 3;

  const tape = $('#tape');
  while (tape.children.length > printed) tape.lastElementChild.remove();
  for (let i = tape.children.length; i < printed; i++){
    const l = TAPE[i], row = document.createElement('div');
    row.className = 'tline' + (l.rule ? ' rule' : '');
    row.style.cssText = `font-size:${l.size}px;color:${l.color};font-weight:${l.weight};letter-spacing:${l.track}em`;
    row.innerHTML = `<span>${l.k}</span><span>${l.v}</span>`;
    tape.appendChild(row);
  }

  $('#printed').textContent = `PRINTED ${printed} / ${TAPE.length} LINES`;
  $('#caret').hidden = !(step >= 4 && !done);
  $('#feeding').classList.toggle('on', step >= 3 && !done);
  $('#stub').classList.toggle('on', done);
  $('#guard').classList.toggle('on', done);
  document.querySelectorAll('.ev').forEach(el=>{
    el.classList.toggle('on', step >= NOTES[Number(el.dataset.i)].at);
  });
}

function run(){
  timers.forEach(clearTimeout); timers = [];
  step = 0; paint();
  const at = (ms, s) => timers.push(setTimeout(()=>{ step = s; paint(); }, ms));
  at(300,1); at(1500,2); at(2700,3);
  TAPE.forEach((_,i)=> at(3300 + i*250, 4+i));
  at(3300 + TAPE.length*250 + 260, TAPE.length + 4);
}

$('#run').addEventListener('click', run);
run();
