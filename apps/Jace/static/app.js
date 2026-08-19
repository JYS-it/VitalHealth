/* CTRSE frontend — Alpine component. Fetch + render ONLY.
 *
 * Core invariant (spec §0): this file computes NO clinical fact. Every level,
 * colour, label, confidence word, probability, threshold_context, escalation_basis,
 * source and guardrail verdict is read straight from the API responses. The ONLY
 * arithmetic is pure presentation on already-supplied facts: a probability -> a bar
 * width, and the supplied threshold -> a tick position. Colour/label/opacity/basis
 * text are *selected* or *formatted* from supplied facts, never derived.
 *
 * Flow: "Clinical Triage" opens the INTAKE (note -> confirm -> result); the result renders in
 * place, in the intake column, so nothing else on screen can overwrite it. The sample-patient
 * browser is a separate inspection surface over 245 precomputed fixtures, not a worklist.
 * There is one result unit (`col`) — the two-column compare view was removed along with its
 * columns array, activeSlot and compareMode.
 */

function makeColumn() {
  return {
    id: null,
    detail: null,
    payload: null,                           // 16-key explain payload for note-derived patients (no id)
    loadingDetail: false,
    // Per-register, not one boolean: the two registers now live in separate columns with their
    // own Generate button, so one generating flag would disable both and skeleton both.
    generating: { justify: false, handover: false },
    gen: { justify: null, handover: null },  // both registers
    genTime: null,                           // wall-clock HH:MM when generation completed
    extras: null,                            // note-derived display facts for the SBAR
                                             // ({allergies, onset, history_mentions,
                                             // medications}) — never model input
    printing: false,                         // marks the hnote as the print target
    copied: false,                           // transient "Copied ✓" feedback
    error: '',
  };
}

// Fresh intake state (§5 two-zone intake -> confirm -> result). Base shape (stage through
// result) now lives in intake-shared.js's makeIntake() — shared with self-check.js's confirm
// screen — extended here with the two fields only the clinician flow uses.
function makeIntake() {
  return {
    ...window.VHIntake.makeIntake(),
    backstop: null,            // /api/extract-guardrail-test result (injection demo)
    logCount: null,
    resultExtras: null,        // display_extras for `result` — carried so re-seating the
                               // finished intake restores the SBAR Background, not just the
                               // model panel
  };
}

document.addEventListener('alpine:init', () => {
  Alpine.data('ctrseApp', () => ({
    // ---- global state ----
    meta: null,
    patients: [],
    levels: ['P1', 'P2', 'P3', 'P4'],
    levelFilter: 'all',
    error: '',
    viewerName: 'guest',
    role: null,                // 'patient' | 'clinician' | null (unauth / standalone dev)

    // ---- surface mode + intake (§5) ----
    mode: 'dashboard',         // 'dashboard' | 'browse' | 'intake'
    activeNav: 'dashboard',    // 'dashboard' | 'triage' | 'results'
    vocab: null,               // { complaints: [...], arrival_modes: [...] } from /api/vocab
    intake: makeIntake(),
    // Demo mock-patient notes (demo_data/characters.py, synced via
    // demo_data/sync_jace_seeds.py). Unpinned: extraction runs live for these.
    //
    // The second row of built-in edge-case seeds (thin note, injection, contradiction,
    // paediatric refusal, …) was removed from the UI. Those notes still exist and are still
    // pinned — prep_pinned_extractions.py owns the canonical list and remains the single
    // source of truth for sample/pinned_extractions.json, which is what tests/test_intake_api.py
    // now reads. They can still be exercised by pasting the note into the box.
    characterSeeds: window.DEMO_CHARACTER_SEEDS || [],

    // The single Model + Gen-AI result unit. Named `col` so the markup's existing `col.*`
    // bindings resolve straight off the component scope now that the compare-mode x-for
    // wrapper (and its second column) are gone.
    col: makeColumn(),
    // Results from this browser session, newest first — /api/predict already persists every
    // assessment to the shared store, so this is for stepping back to something from a minute
    // ago, not an archive.
    sessionAssessments: [],

    // ---- lifecycle ----
    async init() {
      try {
        const meRes = await fetch('/api/me');
        if (meRes.ok) {
          const me = await meRes.json();
          if (me?.authenticated) {
            this.viewerName = me.display_name || me.email || 'user';
            this.role = me.role || null;
          } else {
            this.viewerName = 'guest';
            this.role = null;
          }
        }
      } catch (e) {
        this.viewerName = 'guest';
      }

      // A patient never needs the clinical workspace's own data, and every
      // one of the three calls below 403s for a patient session anyway
      // (require_clinician_for_clinical_api) — skip them so a patient's
      // dashboard load doesn't surface an avoidable error banner for calls
      // it was never going to use. Everything else in init() (the /api/me
      // call above, syncSharedSection below) still runs for both roles.
      if (this.isPatient) {
        this.syncSharedSection();
        return;
      }

      window.addEventListener('afterprint', () => {
        this.col.printing = false;
      });
      try {
        const [metaRes, patRes] = await Promise.all([
          fetch('api/meta'),
          fetch('api/patients'),
        ]);
        this.meta = await metaRes.json();
        const pat = await patRes.json();
        this.levels = pat.levels || this.levels;
        this.patients = pat.patients || [];
      } catch (e) {
        this.error = 'Could not reach the API. Start the server with "uvicorn api:app" and reload.';
      }
      // Controlled vocabulary for the confirm dropdowns — non-fatal if unavailable.
      try {
        this.vocab = await (await fetch('api/vocab')).json();
      } catch (e) { this.vocab = null; }
      this.syncSharedSection();
    },

    syncSharedSection(section = null) {
      if (!document.body || !document.body.dataset) return;
      document.body.dataset.vitalhealthSection = section || this.activeNav || this.mode || 'dashboard';
    },

    // ---- role (§Patient self-check patient/clinician split) ----
    // Mirrors dashboard.js's own isClinician getter but reads app.js's own
    // `role` (captured from /api/me above) rather than dashboard.js's —
    // deliberately not shared state, since the two are separate Alpine
    // roots and dashboard.js's own comment already documents why it keeps
    // its own copy: "Editing `role` in devtools changes nothing about what
    // the API will hand back." The clinical workspace's own gate below is
    // presentation only, same as that one — every route it calls is
    // independently enforced server-side regardless of what isPatient says.
    get isPatient() {
      return this.role === 'patient';
    },
    get isClinician() {
      return this.role === 'clinician';
    },

    // ---- surface mode switch ----
    setMode(m) {
      // Defence in depth: a patient has no clinical workspace to switch
      // into (the workspace itself is gated out of the DOM by isPatient in
      // index.html), so this can only be reached by a stray call.
      if (this.isPatient) return;
      if (m === this.mode) return;
      this.mode = m;
      // Returning to a finished intake re-seats its result. This must carry `extras` as well as
      // detail/payload: extras is where allergies, onset, Hx, Meds, sex and pain score live, so
      // restoring without it silently empties the SBAR Background.
      if (m === 'intake' && this.intake.stage === 'result' && this.intake.result) {
        this.seatResult(this.intake.result, this.intake.resultExtras);
      }
    },

    openDashboard() {
      this.activeNav = 'dashboard';
      this.setMode('dashboard');
      this.syncSharedSection('dashboard');
    },

    // "Clinical Triage" opens the intake — the tool's actual job. It used to open the sample
    // browser, leaving the extractor two clicks deep behind a second nav bar.
    openTriage() {
      this.openIntake();
    },

    openIntake() {
      this.activeNav = 'triage';
      this.setMode('intake');
      this.syncSharedSection('clinical triage intake');
    },

    // The 245 precomputed sample patients — an inspection surface for model behaviour across a
    // cohort, not a live worklist.
    openCohort() {
      this.activeNav = 'triage';
      this.setMode('browse');
      this.syncSharedSection('clinical triage sample cohort');
    },

    welcomeName() {
      const raw = String(this.viewerName || '').trim();

      if (!raw || raw.toLowerCase() === 'guest') {
        return 'guest';
      }

      const emailPrefix = raw.includes('@') ? raw.split('@')[0] : raw;
      const cleaned = emailPrefix.replace(/[._-]+/g, ' ').trim();

      if (!cleaned) {
        return 'user';
      }

      return cleaned
        .split(/\s+/)
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
    },

    get filteredPatients() {
      if (this.levelFilter === 'all') return this.patients;
      return this.patients.filter((p) => p.predicted_level === this.levelFilter);
    },

    isSelectedAnywhere(id) {
      return this.col.id === id;
    },

    // Seat a finished assessment into the result unit. One place, so every caller (a fresh
    // prediction, returning to the intake, picking from the session strip) restores the SAME
    // set of fields — extras included, which is what the old two-line re-seat dropped.
    seatResult(detail, extras = null, gen = null, genTime = null) {
      const col = this.col;
      col.id = null;
      col.detail = detail;
      col.payload = detail?.payload || null;
      col.extras = extras;
      col.gen = gen || { justify: null, handover: null };
      col.genTime = genTime || null;
      col.error = '';
      col.loadingDetail = false;
    },

    async selectPatient(id) {
      const col = this.col;
      col.id = id;
      col.payload = null;
      col.detail = null;
      col.gen = { justify: null, handover: null };
      col.genTime = null;
      col.error = '';
      col.extras = null;
      col.loadingDetail = true;

      try {
        const res = await fetch(`api/patients/${encodeURIComponent(id)}`);
        col.detail = await res.json();
      } catch (e) {
        col.error = 'Could not load the selected patient.';
      } finally {
        col.loadingDetail = false;
      }
    },

    // ONE register per call. The two registers live in separate columns with their own button,
    // so the old both-at-once Promise.all would spend two API calls whenever a clinician wanted
    // one. No full-screen overlay either — each column skeletons itself in place.
    async generate(register) {
      const col = this.col;
      if (!(col.id || col.payload) || col.generating[register]) return;

      col.generating[register] = true;
      col.error = '';

      try {
        const res = await fetch('api/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            use_case: register,
            prefer_live: true,
            ...(col.id ? { patient_id: col.id } : { payload: col.payload }),
          }),
        });
        if (!res.ok) throw new Error(await res.text());
        const result = await res.json();

        // api.py owns the canonical GenAI contract. Adapt its registers to this UI's
        // presentation fields instead of maintaining a second clinical-generation path.
        col.gen[register] = register === 'handover'
          // core._envelope emits `assessment`/`recommendation` for the handover use case
          // (ctrse_core.py's _envelope, uc == "B").
          ? {
              source: result.source,
              assessment: result.assessment,
              recommendation: result.recommendation,
              guardrail: result.guardrails,
              disclaimer: result.disclaimer,
            }
          : {
              source: result.source,
              text: result.text,
              guardrail: result.guardrails,
              disclaimer: result.disclaimer,
            };

        col.genTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        // Write back to the session record this result came from, matched by object identity,
        // so stepping away and returning keeps the registers instead of re-billing for them.
        const rec = this.sessionAssessments.find((r) => r.detail === col.detail);
        if (rec) {
          rec.gen = col.gen;
          rec.genTime = col.genTime;
        }
      } catch (e) {
        col.error = 'Could not generate explanation. Check the API logs and Gen-AI configuration.';
      } finally {
        col.generating[register] = false;
      }
    },

    barWidth(p) {
      return `${Math.max(0, Math.min(1, Number(p || 0))) * 100}%`;
    },
    tickPos() {
      return `${Math.max(0, Math.min(1, Number(this.meta?.p1_threshold || 0))) * 100}%`;
    },
    levelColour(lvl) {
      const p = this.patients.find((x) => x.predicted_level === lvl);
      return p?.level_colour || '#999';
    },
    isPredicted(detail, lvl) {
      return detail?.predicted_level === lvl;
    },
    basisText(detail) {
      if (!detail) return '';
      return detail.escalation_basis || detail.threshold_context || 'model output';
    },
    confCueClass(w) {
      const x = String(w || '').toLowerCase();
      if (x.includes('high')) return 'conf--high';
      if (x.includes('low')) return 'conf--low';
      return 'conf--mid';
    },
    sourceMeta(source) {
      return window.VHIntake.sourceMeta(source);
    },

    // Reads `triage_vitals` — the ONLY shape either data path produces. This previously read
    // detail.hr / .sbp / .rr / .o2 / .temp, which neither GET /api/patients/{id} (api.py) nor
    // predict_from_fields has ever returned, so it produced [] for every patient and the SBAR
    // Background, Copy and Print all reported "none recorded" regardless of what was measured.
    // Abnormality comes from each entry's own `status`, not from string-matching
    // `abnormal_vitals` — those entries are lowercase ("hr=121") and the old check tested
    // startsWith('HR='), so the emphasis never fired either.
    vitalsRow(detail) {
      const tv = detail?.triage_vitals || {};
      const rows = [];
      const seg = (key, label, text) => {
        const v = tv[key];
        if (!v || v.value === null || v.value === undefined) return;
        rows.push({ label, text: text ?? String(v.value), abn: v.status && v.status !== 'normal' });
      };
      seg('hr', 'HR');
      // BP is one clinical reading; render it as a pair and treat either half being out of
      // range as abnormal.
      if (tv.sbp || tv.dbp) {
        rows.push({
          label: 'BP',
          text: `${tv.sbp?.value ?? '—'}/${tv.dbp?.value ?? '—'}`,
          abn: (tv.sbp?.status && tv.sbp.status !== 'normal') ||
               (tv.dbp?.status && tv.dbp.status !== 'normal'),
        });
      }
      seg('rr', 'RR');
      // triage_vitals.o2 carries its own decoded `device` (RA / O2) — a saturation without the
      // device it was measured on is not a complete handover fact.
      seg('o2', 'SpO₂', tv.o2 ? `${tv.o2.value}%${tv.o2.device ? ` ${tv.o2.device}` : ''}` : null);
      seg('temp', 'Temp', tv.temp ? `${tv.temp.value}°C` : null);
      return rows;
    },

    // utilisation_history in words rather than the raw "edvisits: 3" key dump this used to
    // emit. It only ever carries n_edvisits / n_admissions / n_surgeries, and only when the
    // count is non-zero, so a fixed noun map covers it; an unexpected key keeps its own name
    // rather than being dropped.
    edLine(detail) {
      const NOUN = { n_edvisits: 'prior ED visit', n_admissions: 'prior admission', n_surgeries: 'prior surgery' };
      const PLURAL = { n_surgeries: 'prior surgeries' };
      return Object.entries(detail?.utilisation_history || {})
        .filter(([, v]) => Number(v) > 0)
        .map(([k, v]) => {
          const n = Number(v);
          const noun = NOUN[k] || k.replace('n_', '');
          return `${n} ${n === 1 ? noun : (PLURAL[k] || `${noun}s`)}`;
        })
        .join(' · ');
    },

    // high_importance_features_present as plain clinical phrases. Prefix-mapped, never invented:
    // an unrecognised token keeps its own name rather than acquiring a label nobody verified.
    //
    // cc_ drivers are dropped rather than rendered. The frontend holds no complaint-label map
    // (PATIENT_COMPLAINT_LABELS lives in core and is not exposed over the API), so a cc_ driver
    // could only be printed as the raw unspaced token — and it would say nothing the Situation
    // block's complaint line and "Acuity driven by" line do not already say.
    driverWords(detail) {
      const VITAL = { hr: 'HR', sbp: 'SBP', dbp: 'DBP', rr: 'RR', o2: 'SpO₂', temp: 'temp', o2_device: 'O₂ device' };
      const spaced = (s) => String(s).replace(/_/g, ' ').trim();
      return (detail?.high_importance_features_present || []).filter((f) => !String(f).startsWith('cc_')).map((f) => {
        const t = String(f);
        if (t.startsWith('hx_')) return `hx of ${spaced(t.slice(3))}`;
        if (t.startsWith('med_')) return `${spaced(t.slice(4))} meds at home`;
        if (t.startsWith('triage_vital_')) {
          const v = t.slice('triage_vital_'.length);
          return `recorded ${VITAL[v] || spaced(v)}`;
        }
        if (t === 'arrivalmode') return 'arrival mode';
        if (t === 'dep_name') return 'department';
        return spaced(t);
      });
    },

    _displayExtras(col) {
      return col?.extras || col?.payload?.display_extras || col?.detail?.display_extras || {};
    },
    allergyLine(col) {
      const extras = this._displayExtras(col);
      const allergies = extras.allergies || [];
      const vals = allergies
        .map((a) => String(a?.value ?? a ?? '').trim())
        .filter(Boolean);
      return vals.length ? vals.join(', ') : 'not stated';
    },
    onsetFor(col, complaint) {
      if (!complaint) return '';
      const extras = this._displayExtras(col);
      const onset = extras.onset || [];
      const item = onset.find((o) => o.complaint === complaint);
      return item?.value || '';
    },
    historyLine(col) {
      const items = this._displayExtras(col).history_mentions || [];
      return items.map((h) => String(h?.text ?? '').trim()).filter(Boolean).join(' · ');
    },
    sexLine(col) {
      return this._displayExtras(col).sex || '';
    },
    painLine(col) {
      const p = this._displayExtras(col).pain_score;
      return p && p.value !== null && p.value !== undefined ? `${p.value}/10` : '';
    },
    // Onset for every coded complaint, not only the first — a handover that drops the timing of
    // a second complaint has dropped a fact the receiving clinician needs.
    complaintLine(col) {
      const cc = col?.detail?.active_chief_complaints || [];
      return cc.map((c) => {
        const onset = this.onsetFor(col, c);
        return onset ? `${c} (${onset})` : c;
      }).join(' · ');
    },
    // Why this acuity, in words, straight from the code-owned escalation_basis — so the
    // Situation block answers it without the reader parsing the LLM Assessment.
    basisLine(detail) {
      if (!detail) return '';
      if (detail.red_flag_triggered) {
        return `red-flag override on ${detail.red_flag_complaint || 'a red-flag complaint'} — floored to P1`;
      }
      return {
        protocol: 'protocol-based triage priority, not physiological instability',
        physiology: 'out-of-range recorded vitals',
        complaint: 'the coded chief complaint and its historical acuity',
        mixed: 'several recorded factors together',
        routine: 'routine — no escalating factor recorded',
        model_level: 'the model level, with no rule-based escalation',
      }[detail.escalation_basis] || '';
    },
    medicationsLine(col) {
      const items = this._displayExtras(col).medications || [];
      return items.map((m) => String(m?.text ?? '').trim()).filter(Boolean).join(' · ');
    },
    // The clinician's free text, verbatim. Rendered with x-text, so it is inert markup-wise,
    // and it is display-only — see the comment where it is put on display_extras.
    noteText(col) {
      return this._displayExtras(col).note || '';
    },
    // Missingness stated once, in one place, as a finding. "Never recorded" and "asked and
    // answered none" are different facts, and a handover that blurs them is worse than one
    // that stays silent — so each clause only fires where the distinction is actually known.
    gapsLine(col) {
      const d = col?.detail || {};
      const extras = this._displayExtras(col);
      const gaps = [];
      const missing = d.vitals_not_recorded || [];
      if (missing.length) gaps.push(`${missing.length} of 6 vitals not recorded (${missing.join(', ')})`);
      const noOnset = (d.active_chief_complaints || []).filter((c) => !this.onsetFor(col, c));
      if (noOnset.length) gaps.push(`onset not documented for ${noOnset.join(', ')}`);
      // Both of these keys exist on the intake path even when empty. The 245 browsed sample
      // rows carry no display_extras at all, so the keys are absent there and these stay
      // quiet — never collected is not the same claim as never stated.
      if (extras.allergies && !extras.allergies.length) gaps.push('allergy status not stated');
      if ('pain_score' in extras && !extras.pain_score) gaps.push('no pain score recorded');
      return gaps.join(' · ');
    },
    // The Recommendation is now asked for as one item per line. Returns [] for a single
    // unbroken line so the template falls back to a paragraph instead of rendering a one-item
    // list — the stale pinned records and any model that ignores the shape still read fine.
    recLines(col) {
      const lines = String(col?.gen?.handover?.recommendation || '')
        .split('\n')
        .map((l) => l.replace(/^\s*[-•*]\s*/, '').trim())
        .filter(Boolean);
      return lines.length > 1 ? lines : [];
    },

    printHandover() {
      const col = this.col;
      col.printing = true;
      this.$nextTick(() => window.print());
    },
    async copyHandover() {
      const col = this.col;
      const text = this.handoverText(col);
      try {
        await navigator.clipboard.writeText(text);
        col.copied = true;
        setTimeout(() => { col.copied = false; }, 1200);
      } catch (e) {
        col.error = 'Could not copy handover text.';
      }
    },
    handoverText(col) {
      const d = col.detail || {};
      const h = col.gen.handover || {};
      const vitals = this.vitalsRow(d).map((v) => `${v.label} ${v.text}`).join('; ') || 'none recorded';
      const history = this.historyLine(col);
      const medications = this.medicationsLine(col);
      const sex = this.sexLine(col);
      const pain = this.painLine(col);
      const basis = this.basisLine(d);
      const notRecorded = (d.vitals_not_recorded || []).join(', ');
      const ed = this.edLine(d);
      const note = this.noteText(col);
      const drivers = this.driverWords(d).join(' · ');
      const gaps = this.gapsLine(col);
      const rec = this.recLines(col);
      // Built section by section so an unfilled conditional line can be dropped without also
      // dropping the blank separators between sections. Mirrors the on-screen note, so a pasted
      // handover and a printed one carry the same facts.
      const section = (lines) => lines.filter(Boolean).join('\n');
      return [
        'CTRSE TRIAGE HANDOVER',
        section([
          'S:',
          `  Priority: ${d.predicted_level || ''} · ${d.level_label || ''} (confidence: ${d.confidence_word || 'not stated'})`,
          `  Age: ${d.age ?? 'not stated'}${sex ? ` · Sex: ${sex}` : ''}`,
          `  Complaint: ${this.complaintLine(col) || 'none recorded'}`,
          `  Arrival: ${d.arrival_mode || 'not stated'}`,
          basis ? `  Acuity driven by: ${basis}` : '',
          // Indented as a block so a multi-line note stays visibly part of S when pasted.
          note ? `  Triage note (verbatim):\n${note.split('\n').map((l) => `    ${l}`).join('\n')}` : '',
        ]),
        section([
          'B:',
          `  Allergies: ${this.allergyLine(col)}`,
          pain ? `  Pain score: ${pain}` : '',
          history ? `  Hx: ${history}` : '',
          medications ? `  Meds: ${medications}` : '',
          ed ? `  ED history: ${ed}` : '',
          drivers ? `  Model drivers: ${drivers}` : '',
          gaps ? `  Information gaps: ${gaps}` : '',
        ]),
        // Vitals open A, not B — same order as the rendered note, so a pasted handover and the
        // screen carry the same facts in the same places.
        section([
          'A:',
          `  Triage vitals: ${vitals}`,
          notRecorded ? `  Not recorded: ${notRecorded}` : '',
          `  ${h.assessment || h.text || ''}`,
        ]),
        rec.length
          ? section(['R:', ...rec.map((r) => `  - ${r}`)])
          : `R: ${h.recommendation || ''}`,
        h.disclaimer || '',
      ].filter(Boolean).join('\n\n');
    },

    // combinedGuard() (the "Guardrail: pass / review" chip) and guardFlags() (the per-register
    // flag list) were removed along with the "Test guardrail" control and the two "Draft —
    // requires clinician review" tags. The guardrail still RUNS on every generation and its
    // verdict still travels in result.guardrails — it simply has no rendered surface on the
    // clinician page any more. `col.gen[register].guardrail` is still populated for anyone
    // who wants to put one back.

    backgroundDrivers(detail) {
      const rows = [];
      if (!detail) return rows;
      if (detail.red_flag_triggered) rows.push('red-flag override');
      if (detail.threshold_sensitive) rows.push('threshold-sensitive');
      if ((detail.abnormal_vitals || []).length) rows.push('abnormal vitals');
      if ((detail.high_importance_features_present || []).length) rows.push('model driver features present');
      return rows;
    },

    seedCharacter(i) {
      const s = this.characterSeeds[i];
      if (!s) return;
      this.intake.note = s.note || s.text || '';
      this.intake.stage = 'input';
      this.intake.extractError = '';
      if (s.vitals) {
        this.intake.vitals = {
          ...this.intake.vitals,
          ...s.vitals,
        };
        this.intake.noVitals = false;
      }
    },

    async extract() {
      this.intake.extracting = true;
      this.intake.extractError = '';
      this.intake.extraction = null;
      this.intake.fields = null;
      this.intake.refusal = null;
      this.intake.flagsAck = false;
      this.intake.backstop = null;
      window.VitalHealthLoading?.show('Extracting clinical details');

      try {
        const res = await fetch('api/extract', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            note: this.intake.note,
            vitals: this.intake.noVitals ? null : this.intake.vitals,
            no_vitals: this.intake.noVitals,
          }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'extract failed');

        this.intake.extraction = data;
        this.intake.refusal = data.refusal_reason || null;
        this.intake.fields = this.buildConfirmFields(data);
        this.intake.stage = 'confirm';
      } catch (e) {
        this.intake.extractError = 'Could not extract fields. Check the API logs.';
      } finally {
        this.intake.extracting = false;
        window.VitalHealthLoading?.hide();
      }
    },

    // ---- intake-shared.js delegates (window.VHIntake) ----
    // These keep their original method names so every existing index.html
    // binding (x-text="highlightedNote()", @click, etc.) is untouched; the
    // logic itself now lives in intake-shared.js so self-check.js can call
    // the identical functions instead of maintaining its own copies.
    buildConfirmFields(extraction) {
      return window.VHIntake.buildConfirmFields(extraction);
    },

    highlightedNote() {
      // Spans are validated against — and must be highlighted against — the PREPARED note
      // (core.py's _prepare_note docstring: "Spans are validated against — and the UI
      // highlights — this prepared text"), not the raw textarea value: redaction can shift or
      // remove text a span would otherwise match. Falls back to the raw note only before any
      // extraction has run (e.g. while still on stage 1).
      const note = this.intake.extraction?.note_used ?? this.intake.note;
      return window.VHIntake.highlightedNote(note, this.intake.hoverSpan);
    },

    escapeHtml(s) {
      return window.VHIntake.escapeHtml(s);
    },

    friendlyFlag(f) {
      return window.VHIntake.friendlyFlag(f, 'clinician');
    },

    async runExtractBackstop() {
      this.intake.backstop = null;
      try {
        const res = await fetch('api/extract-guardrail-test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });
        this.intake.backstop = await res.json();
      } catch (e) {
        this.intake.backstop = { synthetic: 'Could not run backstop test.', flags: [] };
      }
    },

    extractSourceMeta() {
      const src = this.intake.extraction?.source || '';
      return this.sourceMeta(src);
    },

    backstopSummary() {
      const b = this.intake.backstop;
      if (!b) return '';
      return `${b.synthetic || 'synthetic test'} · ${(b.flags || []).join(', ') || 'no flags'}`;
    },

    get allAcked() {
      return window.VHIntake.allAcked(this.intake.fields, this.intake.extraction, this.intake.flagsAck);
    },

    _num(v) {
      return window.VHIntake.num(v);
    },

    _confirmedRequest() {
      const f = this.intake.fields;
      const oxygenDeviceCode = { RA: 0, O2: 1 }[this.intake.vitals.device] ?? null;
      const complaints = (f.complaints || [])
        .map((c) => ({
          token: String(c.token || '').trim(),
          evidence: c.evidence || null,
        }))
        .filter((c) => c.token)
        .slice(0, 2);

      return {
        note: this.intake.note,
        age: this._num(f.age.value),
        sex: f.sex.value || null,
        arrival_mode: f.arrival.value || null,
        complaints,
        vitals: this.intake.noVitals ? null : {
          hr: this._num(this.intake.vitals.hr),
          sbp: this._num(this.intake.vitals.sbp),
          dbp: this._num(this.intake.vitals.dbp),
          rr: this._num(this.intake.vitals.rr),
          o2: this._num(this.intake.vitals.o2),
          o2_device: oxygenDeviceCode,
          temp: this._num(this.intake.vitals.temp),
          temp_unit: this.intake.vitals.temp_unit || 'C',
        },
        no_vitals: this.intake.noVitals,
        extraction: this.intake.extraction,
        display_extras: {
          allergies: f.allergies || [],
          onset: this.intake.extraction?.onset || [],
          history_mentions: this.intake.extraction?.history_mentions || [],
          medications: this.intake.extraction?.medications || [],
          // Handover identifier only. Deliberately NOT added to explain()'s payload — `sex` is
          // genuinely absent from it, and widening that dict would break test_parity.py's
          // shipped-payload equality. Consequence: sex shows for note-derived patients and is
          // simply absent for the 245 browsed sample patients, which carry no display_extras.
          sex: f.sex?.value || null,
          pain_score: this.intake.extraction?.pain_score || null,
          // The clinician's own free text, rendered verbatim under S. DISPLAY ONLY — it must
          // never reach the handover prompt. The note already goes to the extraction model
          // under span-or-silence, where every value it yields must quote a literal substring;
          // feeding it to the clinician register instead would let text inside the note steer
          // prose a clinician reads as fact, with no such check on the way out.
          note: this.intake.note || '',
        },
      };
    },

    async confirmRun() {
      if (!this.allAcked || this.intake.predicting) return;

      this.intake.predicting = true;
      this.intake.predictError = '';

      try {
        const confirmed = this._confirmedRequest();
        const res = await fetch('api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            age: confirmed.age,
            sex: confirmed.sex,
            arrival_mode: confirmed.arrival_mode,
            complaints: confirmed.complaints,
            vitals: confirmed.vitals ? {
              hr: confirmed.vitals.hr,
              sbp: confirmed.vitals.sbp,
              dbp: confirmed.vitals.dbp,
              rr: confirmed.vitals.rr,
              o2: confirmed.vitals.o2,
              temp: confirmed.vitals.temp,
              temp_unit: confirmed.vitals.temp_unit,
            } : null,
          }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'predict failed');

        const extras = data.display_extras || this._confirmedRequest().display_extras;
        this.intake.result = data;
        this.intake.resultExtras = extras;
        this.intake.logCount = data.log_count ?? null;
        this.intake.stage = 'result';

        this.seatResult(data, extras);
        this.recordAssessment(data, extras);
        // Deliberately NOT switching mode. This used to call openResults(), which put the
        // finished assessment beside the sample-patient sidebar — where the next click ran
        // selectPatient() and overwrote it. Staying in intake mode keeps the result in the
        // intake column with its own "New triage" toolbar visible.
      } catch (e) {
        this.intake.predictError = 'Could not run triage prediction. Check the API logs.';
      } finally {
        this.intake.predicting = false;
      }
    },

    vitalsSummary() {
      return window.VHIntake.vitalsSummary(this.intake.vitals, this.intake.noVitals);
    },

    backToInput() {
      this.intake.stage = 'input';
    },

    resetIntake() {
      this.intake = makeIntake();
      this.col = makeColumn();
      this.openIntake();
    },

    // ---- session assessments ----
    recordAssessment(detail, extras) {
      this.sessionAssessments.unshift({
        key: `${Date.now()}-${this.sessionAssessments.length}`,
        at: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        level: detail?.predicted_level || '',
        colour: detail?.level_colour || 'var(--accent)',
        summary: (detail?.active_chief_complaints || []).join(', ') || 'no complaint coded',
        detail,
        extras,
        // Filled in by generate() when the registers are produced for this assessment, so
        // returning to it does not silently drop Gen-AI output already paid for.
        gen: null,
        genTime: null,
      });
      this.sessionAssessments = this.sessionAssessments.slice(0, 8);
    },

    openAssessment(key) {
      const rec = this.sessionAssessments.find((r) => r.key === key);
      if (!rec) return;
      this.seatResult(rec.detail, rec.extras, rec.gen, rec.genTime);
      this.intake.result = rec.detail;
      this.intake.resultExtras = rec.extras;
      this.intake.stage = 'result';
      this.openIntake();
    },
  }));
});
