/* ════════════════════════════════════════
   NeuroSonic AI  —  recorder.js
   Handles: mic recording, waveform anim,
   API call to /predict-cough/, UI updates
   ════════════════════════════════════════ */

// ── STATE ──────────────────────────────
let recorder   = null;
let chunks     = [];
let stream     = null;
let isRecording = false;

let sessionCount  = 0;
let confidenceSum = 0;
const history     = [];

// ── WAVEFORM SETUP ─────────────────────
const waveformEl = document.getElementById('waveform');
const BAR_COUNT  = 36;
const wbars      = [];

(function initWaveform() {
  for (let i = 0; i < BAR_COUNT; i++) {
    const b = document.createElement('div');
    b.className = 'wbar';
    const h   = 10 + Math.random() * 36;
    const dur = 0.7 + Math.random() * 0.8;
    b.style.setProperty('--h',     h + 'px');
    b.style.setProperty('--dur',   dur + 's');
    b.style.setProperty('--delay', (i * 0.035) + 's');
    waveformEl.appendChild(b);
    wbars.push(b);
  }
})();

let waveInterval = null;

function startWave() {
  wbars.forEach(b => b.classList.add('active'));
  waveInterval = setInterval(() => {
    wbars.forEach(b => {
      const h = 8 + Math.random() * 40;
      b.style.setProperty('--h', h + 'px');
    });
  }, 350);
}

function stopWave() {
  wbars.forEach(b => b.classList.remove('active'));
  clearInterval(waveInterval);
  waveInterval = null;
}

// ── RECORDING ──────────────────────────
function startRecording() {
  if (isRecording) return;

  navigator.mediaDevices.getUserMedia({ audio: true })
    .then(s => {
      stream    = s;
      recorder  = new MediaRecorder(s);
      chunks    = [];
      isRecording = true;

      recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.start(100);

      setUIRecording(true);
      startWave();
    })
    .catch(err => {
      console.error('Microphone error:', err);
      showError('Microphone access denied. Please allow mic permissions.');
    });
}

function stopRecording() {
  if (!isRecording || !recorder) return;
  isRecording = false;

  recorder.stop();
  stream.getTracks().forEach(t => t.stop());

  setUIAnalysing();
  stopWave();

  recorder.onstop = () => {
    const blob     = new Blob(chunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', blob, 'cough.wav');

    fetch('/predict-cough/', {
      method: 'POST',
      body: formData,
      headers: { 'X-CSRFToken': getCookie('csrftoken') }
    })
    .then(res => {
      if (!res.ok) throw new Error('Server error ' + res.status);
      return res.json();
    })
    .then(data => {
      renderResult(data);
      setUIIdle();
    })
    .catch(err => {
      console.error('Prediction error:', err);
      showError('Analysis failed — please try again.');
      setUIIdle();
    });
  };
}

// ── UI STATE HELPERS ───────────────────
function setUIRecording(on) {
  const btnRecord = document.getElementById('btnRecord');
  const btnStop   = document.getElementById('btnStop');
  const ring      = document.getElementById('ring');
  const ringText  = document.getElementById('ringText');
  const detectSub = document.getElementById('detectSub');
  const sysDot    = document.getElementById('sysDot');
  const sysStatus = document.getElementById('sysStatus');
  const logoPulse = document.getElementById('logoPulse');

  btnRecord.disabled = true;
  btnStop.disabled   = false;
  ring.classList.add('recording');
  ringText.textContent = 'Listening…';
  detectSub.innerHTML  = 'Recording in progress.<br>Tap stop when done.';
  sysDot.className     = 'sys-dot ok';
  sysStatus.textContent = 'Recording…';

  // Hero card
  setHeroState('Listening…', 0, 'var(--green)', 'Capturing audio input');
  animateRingProgress(1.0);
}

function setUIAnalysing() {
  const ringText  = document.getElementById('ringText');
  const detectSub = document.getElementById('detectSub');
  const sysStatus = document.getElementById('sysStatus');
  const sysDot    = document.getElementById('sysDot');

  ringText.textContent = 'Analysing…';
  detectSub.innerHTML  = 'Processing audio with<br>the AI model…';
  sysDot.className     = 'sys-dot warn';
  sysStatus.textContent = 'Analysing…';
  setHeroState('Analysing…', 60, 'var(--amber)', 'Running AI prediction');
}

function setUIIdle() {
  const btnRecord = document.getElementById('btnRecord');
  const btnStop   = document.getElementById('btnStop');
  const ring      = document.getElementById('ring');
  const ringText  = document.getElementById('ringText');
  const detectSub = document.getElementById('detectSub');
  const sysDot    = document.getElementById('sysDot');
  const sysStatus = document.getElementById('sysStatus');

  btnRecord.disabled = false;
  btnStop.disabled   = true;
  ring.classList.remove('recording');
  ringText.textContent = 'Done';
  detectSub.innerHTML  = 'Record again to run<br>another analysis.';
  sysDot.className     = 'sys-dot';
  sysStatus.textContent = 'Not recording';
  animateRingProgress(0);
}

// ── RING PROGRESS ARC ──────────────────
function animateRingProgress(fraction) {
  const circ = 2 * Math.PI * 60; // r=60
  const offset = circ * (1 - fraction);
  const el = document.getElementById('ringProgress');
  if (el) {
    el.style.strokeDasharray  = circ;
    el.style.strokeDashoffset = offset;
  }
}

// ── HERO CARD ──────────────────────────
function setHeroState(stateText, pct, color, subText) {
  document.getElementById('hcState').textContent  = stateText;
  document.getElementById('hcBar').style.width    = pct + '%';
  document.getElementById('hcBar').style.background = color;
  document.getElementById('hcDot').style.background = color;
  document.getElementById('hcSub').textContent    = subText;
}

// ── RENDER RESULT ──────────────────────
function renderResult(data) {
  // Normalise API response fields
  const raw        = data.prediction ?? data.label ?? 'Unknown';
  const confidence = parseFloat(data.confidence ?? data.score ?? 0);
  const pct        = Math.round(confidence * 100);

  // Classify
  const isCough   = /cough/i.test(raw);
  const isNoCough = /no.?cough|normal|clear/i.test(raw);
  let tagClass, dotColor, heroColor, heroState, heroSub;

  if (isCough) {
    tagClass  = 'tag-cough';
    dotColor  = 'var(--red)';
    heroColor = 'var(--red)';
    heroState = 'Cough detected';
    heroSub   = `Confidence: ${pct}%`;
  } else if (isNoCough) {
    tagClass  = 'tag-no-cough';
    dotColor  = 'var(--green)';
    heroColor = 'var(--green)';
    heroState = 'No cough';
    heroSub   = `Confidence: ${pct}%`;
  } else {
    tagClass  = 'tag-unclear';
    dotColor  = 'var(--amber)';
    heroColor = 'var(--amber)';
    heroState = raw;
    heroSub   = `Confidence: ${pct}%`;
  }

  const confBarColor = isCough ? 'var(--red)' : isNoCough ? 'var(--green)' : 'var(--amber)';

  // Result card
  document.getElementById('resultBody').innerHTML = `
    <div class="result-prediction">${raw}</div>
    <span class="result-tag ${tagClass}">${pct}% confidence</span>
    <div class="conf-label">Confidence score</div>
    <div class="conf-bar-bg">
      <div class="conf-bar-fill" style="width:${pct}%; background:${confBarColor};"></div>
    </div>
    <div class="conf-pct">${pct}%</div>
  `;

  // Hero card
  setHeroState(heroState, pct, heroColor, `Confidence: ${pct}%`);
  animateRingProgress(confidence);
  document.getElementById('ring').classList.remove('recording');
  document.getElementById('ringText').textContent = isCough ? 'Cough!' : 'Clear';

  // Stats
  sessionCount++;
  confidenceSum += pct;
  document.getElementById('statSessions').textContent = sessionCount;
  document.getElementById('statLast').textContent     = raw;
  document.getElementById('statLastConf').textContent = pct + '% confidence';
  document.getElementById('statAvg').textContent      = Math.round(confidenceSum / sessionCount) + '%';

  // History
  addHistory(raw, pct, tagClass, dotColor);
}

// ── HISTORY LIST ───────────────────────
function addHistory(prediction, pct, tagClass, dotColor) {
  const now = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  history.unshift({ prediction, pct, tagClass, dotColor, timeStr });
  if (history.length > 8) history.pop();

  const list = document.getElementById('historyList');
  list.innerHTML = history.map(h => `
    <li class="history-item">
      <div class="hi-left">
        <span class="hi-dot" style="background:${h.dotColor}"></span>
        <span>${h.prediction}</span>
      </div>
      <div style="display:flex;align-items:center;gap:12px;">
        <span class="result-tag ${h.tagClass}" style="margin:0">${h.pct}%</span>
        <span class="hi-time">${h.timeStr}</span>
      </div>
    </li>
  `).join('');
}

// ── ERROR STATE ────────────────────────
function showError(msg) {
  document.getElementById('resultBody').innerHTML = `
    <p style="color:var(--red);font-size:.875rem;">${msg}</p>
  `;
  setHeroState('Error', 0, 'var(--red)', msg);
}

// ── CSRF HELPER ────────────────────────
function getCookie(name) {
  const v = document.cookie.match('(^|;)\\s*' + name + '\\s*=\\s*([^;]+)');
  return v ? v.pop() : '';
}

// ── INIT ───────────────────────────────
document.getElementById('btnStop').disabled = true;
animateRingProgress(0);
