/**
 * self-check.js — patient triage self-check, the SAME three-stage interface
 * (input -> confirm extracted fields -> result) as the clinician intake
 * (app.js), built on the shared helpers in intake-shared.js. The confirm
 * screen is fully editable against the same controlled vocabulary a
 * clinician sees (GET /api/self-check/options mirrors GET /api/vocab) —
 * there is no separate patient-safe picker. What differs from app.js is
 * downstream of the model, not the interface: no clinician-only GenAI
 * (justification/handover) — instead §Describe-help (use case D, on the
 * confirm screen) and §Patient guidance (use case C, on the result screen).
 * Computes no clinical fact of its own, same invariant as app.js's header
 * comment: every level/colour/label/guidance sentence is read straight from
 * the API responses.
 */
function selfCheckApp() {
  return {
    error: '',

    emergencyNumber: '995',
    crisisNumber: '',
    crisisLabel: '',
    scopeNote: '',
    vocab: null,               // { complaints, arrival_modes } from /api/self-check/options
    // Demo seeds for this page's own extractor, written in first person — the clinician's
    // seeds/character notes (app.js, demo_character_seeds.js) are third-person clinical
    // narration, the wrong register for someone describing their own symptoms. Each carries
    // vitals so one click fills both zones, same affordance as index.html's seed row.
    seeds: [
      { label: 'chest pain', note: 'I have had tight pain in my chest for 30 minutes and feel short of breath.',
        vitals: { hr: 102, sbp: 150, dbp: 92, rr: 20, o2: 95, device: 'RA', temp: 37.0, temp_unit: 'C' } },
      { label: 'headache', note: 'I have had a bad headache since this morning, worse behind my eyes.',
        vitals: { hr: 78, sbp: 118, dbp: 76, rr: 16, o2: 99, device: 'RA', temp: 36.9, temp_unit: 'C' } },
      { label: 'fall', note: 'I fell at home this morning and my hip hurts, I am on blood thinners.',
        vitals: { hr: 90, sbp: 138, dbp: 84, rr: 18, o2: 97, device: 'RA', temp: 36.8, temp_unit: 'C' } },
      { label: 'stomach pain', note: 'I have had stomach pain since last night, it is worse after eating.',
        vitals: { hr: 88, sbp: 128, dbp: 80, rr: 18, o2: 98, device: 'RA', temp: 37.4, temp_unit: 'C' } },
      { label: 'breathless', note: "I can't catch my breath since this morning and I feel dizzy.",
        vitals: { hr: 112, sbp: 148, dbp: 90, rr: 26, o2: 91, device: 'RA', temp: 37.2, temp_unit: 'C' } },
      { label: 'thin note', note: 'I feel unwell.', vitals: null },
    ],

    intake: {
      ...window.VHIntake.makeIntake(),
      describeHelp: null,        // §Describe-help (use case D) — on-demand confirm-screen panel
      describeHelpLoading: false,
      followUp: '',               // the answer text box under describeHelp
      followUpOffset: null,       // where the follow-up begins in extraction.note_used
      guidance: null,             // §Patient guidance (use case C) — result-screen suggestion
      guidanceLoading: false,
    },

    async init() {
      try {
        const res = await fetch('api/self-check/options');
        if (!res.ok) throw new Error('options request failed');
        const data = await res.json();
        const contacts = data.emergency_contacts || {};
        // Fall back to the same number rendered in the banner so a failed
        // request never removes emergency guidance from the page.
        this.emergencyNumber = contacts.emergency_number || this.emergencyNumber;
        this.crisisNumber = contacts.crisis_number || '';
        this.crisisLabel = contacts.crisis_label || '';
        this.scopeNote = data.scope_note || '';
        this.vocab = { complaints: data.complaints || [], arrival_modes: data.arrival_modes || [] };
      } catch (e) {
        this.error = 'Some information on this page could not load. The emergency numbers above are still correct.';
      }
    },

    bandColour(tone) {
      return { critical: 'var(--p1)', warning: 'var(--warn)', neutral: 'var(--accent)' }[tone] || 'var(--accent)';
    },

    // ---- intake-shared.js delegates (window.VHIntake) — same names as app.js's own,
    // register: 'patient' picks the plain-language guardrail-flag wording. ----
    buildConfirmFields(extraction) {
      return window.VHIntake.buildConfirmFields(extraction);
    },
    highlightedNote() {
      // Spans are validated against — and must be highlighted against — the PREPARED note
      // (ctrse_core.py's _prepare_note docstring), not the raw textarea value: redaction can
      // shift or remove text a span would otherwise match. Falls back to the raw note only
      // before any extraction has run (e.g. while still on stage 1). followUpOffset (from the
      // last /api/self-check/extract response) marks where a §Describe-help answer was
      // appended, so it renders visually distinct from the original description.
      const note = this.intake.extraction?.note_used ?? this.intake.note;
      return window.VHIntake.highlightedNote(note, this.intake.hoverSpan, this.intake.followUpOffset);
    },
    escapeHtml(s) {
      return window.VHIntake.escapeHtml(s);
    },
    friendlyFlag(f) {
      return window.VHIntake.friendlyFlag(f, 'patient');
    },
    // forced_ambiguity has no patient-register wording (friendlyFlag falls through to the raw
    // flag name) and duplicates the affected symptom's own "we weren't sure" row — dropped from
    // the notices list here, not from friendlyFlag() itself, since the clinician page still
    // wants to see it verbatim.
    visibleGuardrailFlags() {
      return (this.intake.extraction?.guardrail_flags || [])
        .filter((f) => !String(f).startsWith('forced_ambiguity'));
    },
    get allAcked() {
      // Gate on the same filtered list the notices box actually shows — otherwise a
      // forced_ambiguity-only extraction would show no checkbox to check yet still block
      // confirm forever (VHIntake.allAcked gates on raw guardrail_flags.length).
      const extraction = this.intake.extraction
        ? { ...this.intake.extraction, guardrail_flags: this.visibleGuardrailFlags() }
        : this.intake.extraction;
      return window.VHIntake.allAcked(this.intake.fields, extraction, this.intake.flagsAck);
    },
    vitalsSummary() {
      return window.VHIntake.vitalsSummary(this.intake.vitals, this.intake.noVitals);
    },

    // §Patient guidance (use case C) is prompted for exactly two flowing-prose paragraphs,
    // separated by one blank line (see SYSTEM_PROMPT_PATIENT rule 9) — split here rather than
    // rendered as one <p>, which would collapse the break and read as a single dense block.
    guidanceParagraphs() {
      const text = this.intake.guidance?.text || '';
      return text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
    },

    // Band-aware in place of a flat "What to do" — reuses the urgency band the result screen
    // already has (no new hardcoded copy per band elsewhere), shown for both the loading
    // skeleton and the loaded suggestion so the title doesn't change mid-load.
    guidanceTitle() {
      const band = this.intake.result?.urgency?.band;
      if (band === 'EMERGENCY_NOW') return 'What to do right now';
      if (band === 'URGENT_TODAY') return "What to do before you're seen";
      return 'What to do while you wait';
    },

    seed(i) {
      const s = this.seeds[i];
      if (!s) return;
      this.intake.note = s.note;
      this.intake.stage = 'input';
      this.intake.extractError = '';
      if (s.vitals) {
        this.intake.vitals = { ...this.intake.vitals, ...s.vitals };
        this.intake.noVitals = false;
      } else {
        this.intake.noVitals = true;
      }
    },

    // ---- stage 1 -> 2: extract ----
    async extract() {
      this.intake.extracting = true;
      this.intake.extractError = '';
      this.intake.extraction = null;
      this.intake.fields = null;
      this.intake.refusal = null;
      this.intake.flagsAck = false;
      this.intake.describeHelp = null;
      this.intake.followUp = '';
      this.intake.followUpOffset = null;
      window.VitalHealthLoading?.show('Reading your description');

      try {
        const res = await fetch('api/self-check/extract', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ note: this.intake.note }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'extract failed');

        this.intake.extraction = data;
        this.intake.refusal = data.refusal_reason || null;
        this.intake.fields = this.buildConfirmFields(data);
        this.intake.stage = 'confirm';
        // §Describe-help is on-demand from here (a button on the confirm screen) — see
        // loadDescribeHelp() — not fired automatically, since its answer box needs the patient
        // to actually be looking at the confirm screen first.
      } catch (e) {
        this.intake.extractError = 'Could not read your description. Please try again.';
      } finally {
        this.intake.extracting = false;
        window.VitalHealthLoading?.hide();
      }
    },

    async loadDescribeHelp() {
      // §Describe-help (use case D): coaching grounded in what the extractor found, before
      // any prediction exists. Triggered on demand by a confirm-screen button. Never retried;
      // any failure (network, non-200, guardrail-rejected) just leaves `describeHelp` null and
      // the panel doesn't appear — same silent-degrade posture as loadGuidance() below.
      this.intake.describeHelpLoading = true;
      try {
        const res = await fetch('api/self-check/describe-help', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ extraction: this.intake.extraction }),
        });
        if (!res.ok) {
          this.intake.describeHelp = null;
          return;
        }
        const data = await res.json();
        this.intake.describeHelp = data.available ? data : null;
      } catch (e) {
        this.intake.describeHelp = null;
      } finally {
        this.intake.describeHelpLoading = false;
      }
    },

    async submitFollowUp() {
      // The answer to §Describe-help's question. Sent as its own field alongside the original
      // note (not appended by the browser) — api.py composes them server-side and re-runs the
      // same guarded extractor, so span-or-silence still validates against exactly what was
      // sent, and the confirm screen refreshes with whatever the extractor now finds (e.g. an
      // onset it previously missed).
      const followUp = (this.intake.followUp || '').trim();
      if (!followUp || this.intake.extracting) return;

      this.intake.extracting = true;
      this.intake.extractError = '';
      window.VitalHealthLoading?.show('Reading your description');

      try {
        const res = await fetch('api/self-check/extract', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ note: this.intake.note, follow_up: followUp }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'extract failed');

        this.intake.extraction = data;
        this.intake.refusal = data.refusal_reason || null;
        this.intake.fields = this.buildConfirmFields(data);
        this.intake.followUpOffset = data.follow_up_offset ?? null;
        this.intake.followUp = '';
        // The question this answered no longer applies to the now-updated description — the
        // patient can ask for fresh feedback again if they want it.
        this.intake.describeHelp = null;
      } catch (e) {
        this.intake.extractError = 'Could not add that detail. Please try again.';
      } finally {
        this.intake.extracting = false;
        window.VitalHealthLoading?.hide();
      }
    },

    backToInput() {
      this.intake.stage = 'input';
    },

    // ---- stage 2 -> 3: confirm & run ----
    async confirmRun() {
      if (!this.allAcked || this.intake.predicting) return;

      this.intake.predicting = true;
      this.intake.predictError = '';
      window.VitalHealthLoading?.show('Checking your symptoms');

      try {
        const f = this.intake.fields;
        const oxygenDeviceCode = { RA: 0, O2: 1 }[this.intake.vitals.device] ?? null;
        const complaints = (f.complaints || [])
          .map((c) => ({ token: String(c.token || '').trim(), evidence: c.evidence || null }))
          .filter((c) => c.token)
          .slice(0, 2);

        const res = await fetch('api/self-check', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            age: window.VHIntake.num(f.age.value),
            sex: f.sex.value || null,
            arrival_mode: f.arrival.value || null,
            complaints,
            vitals: this.intake.noVitals ? null : {
              hr: window.VHIntake.num(this.intake.vitals.hr),
              sbp: window.VHIntake.num(this.intake.vitals.sbp),
              dbp: window.VHIntake.num(this.intake.vitals.dbp),
              rr: window.VHIntake.num(this.intake.vitals.rr),
              o2: window.VHIntake.num(this.intake.vitals.o2),
              o2_device: oxygenDeviceCode,
              temp: window.VHIntake.num(this.intake.vitals.temp),
              temp_unit: this.intake.vitals.temp_unit || 'C',
            },
            // The extraction the confirm screen was built from — never re-sent as edited
            // fields, only as the source api.py re-derives red flags from (see api.py's
            // post_self_check comment). Editing fields here can only add to what gets scored,
            // never silently drop a red flag the extractor found.
            extraction: this.intake.extraction,
          }),
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data?.detail || 'Could not run this check. Please try again.');

        this.intake.result = data;
        this.intake.stage = 'result';
        // No prediction exists to explain when the model refused (e.g. out-of-range age) —
        // patient_view's refusal shape doesn't even carry reported_concerns. Skip the call
        // rather than ground §Patient guidance in a mostly-empty payload.
        if (!data.model_refused) {
          this.loadGuidance();
        }
      } catch (e) {
        this.intake.predictError = e.message || 'Could not run this check. Please try again.';
      } finally {
        this.intake.predicting = false;
        window.VitalHealthLoading?.hide();
      }
    },

    async loadGuidance() {
      // §Patient guidance (use case C): a single suggestion covering where/how-soon to seek
      // care, what to do while waiting, and escalation signs — grounded in patient_view PLUS
      // the confirm-screen extraction (allergies/medications/history/onset/note), generated
      // AFTER the deterministic result is already final and shown. Never blocks, never
      // retried, and any failure just leaves `guidance` null.
      this.intake.guidanceLoading = true;
      try {
        const res = await fetch('api/self-check/explain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ patient_view: this.intake.result, extraction: this.intake.extraction }),
        });
        if (!res.ok) {
          this.intake.guidance = null;
          return;
        }
        const data = await res.json();
        this.intake.guidance = data.available ? data : null;
      } catch (e) {
        this.intake.guidance = null;
      } finally {
        this.intake.guidanceLoading = false;
      }
    },

    reset() {
      this.intake = {
        ...window.VHIntake.makeIntake(),
        describeHelp: null,
        describeHelpLoading: false,
        followUp: '',
        followUpOffset: null,
        guidance: null,
        guidanceLoading: false,
      };
      this.error = '';
    },
  };
}
