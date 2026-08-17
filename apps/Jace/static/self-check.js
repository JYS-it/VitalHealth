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
    // Six seeds, one per path through the patient guidance model, so clicking through them shows
    // the generated advice actually CHANGING rather than six variations on the same band. Each
    // states age and sex in the patient's own voice, since the extractor reads them from the note
    // (span-or-silence) rather than from a form field.
    //
    //   stroke      -> EMERGENCY_NOW via the red-flag floor (cc_strokealert)
    //   breathless  -> URGENT_TODAY via the physiology floor (SpO2/RR out of range)
    //   headache    -> SEE_CLINICIAN, routine — nothing escalating recorded
    //   tablets     -> self-care guidance SUPPRESSED (_self_care_suppressed, overdose path)
    //   thin note   -> same band as `headache`, opposite completeness: no onset, no vitals,
    //                  nothing else recorded. The pair is the point — the band matches and the
    //                  guidance does not.
    //   under 18    -> age refusal, handled gracefully
    seeds: [
      { label: 'stroke signs',
        note: "I am a 74 year old man. My face dropped on one side about 40 minutes ago and my "
            + "left arm has gone weak. My speech is slurred. My wife thinks it is a stroke.",
        vitals: { hr: 96, sbp: 178, dbp: 98, rr: 18, o2: 96, device: 'RA', temp: 36.9, temp_unit: 'C' } },
      { label: 'breathless',
        note: "I am a 68 year old woman. I cannot catch my breath since this morning and it is "
            + "worse when I lie down. I have asthma and I use an inhaler.",
        vitals: { hr: 118, sbp: 148, dbp: 90, rr: 28, o2: 89, device: 'RA', temp: 37.6, temp_unit: 'C' } },
      { label: 'headache',
        note: "I am a 29 year old woman. I have had a headache since this morning, mostly behind "
            + "my eyes. It is a 4 out of 10. I am not on any medication and I have no allergies.",
        vitals: { hr: 76, sbp: 116, dbp: 74, rr: 16, o2: 99, device: 'RA', temp: 36.8, temp_unit: 'C' } },
      { label: 'took tablets',
        note: "I am a 22 year old woman. I took a whole box of paracetamol about two hours ago "
            + "because I wanted to hurt myself. I feel sick now.",
        vitals: { hr: 98, sbp: 118, dbp: 72, rr: 18, o2: 98, device: 'RA', temp: 36.7, temp_unit: 'C' } },
      // Deliberately NOT "I feel unwell." — that extracts only the `other` token, which
      // /api/self-check rejects with "we could not identify a symptom", so it demonstrated a
      // validation wall rather than how the tool handles a sparse picture. This version is
      // thin but scorable: one real symptom, no onset, no vitals, nothing else.
      { label: 'thin note',
        note: "I am a 71 year old man. I have been feeling dizzy but I cannot say when it "
            + "started. I have no way to check my blood pressure at home.",
        vitals: null },
      { label: 'under 18',
        note: "I am a 16 year old boy. I fell off my skateboard an hour ago and my wrist hurts, "
            + "about a 6 out of 10.",
        vitals: { hr: 88, sbp: 110, dbp: 70, rr: 18, o2: 98, device: 'RA', temp: 37.0, temp_unit: 'C' } },
    ],

    intake: {
      ...window.VHIntake.makeIntake(),
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
      // before any extraction has run (e.g. while still on stage 1).
      const note = this.intake.extraction?.note_used ?? this.intake.note;
      return window.VHIntake.highlightedNote(note, this.intake.hoverSpan);
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
      // No full-screen overlay here any more: the input card hides and an in-place skeleton
      // shaped like the confirm screen takes its place (self-check.html, "STAGE 1 -> 2
      // skeleton"), so the page transitions into its next state instead of being covered.

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
      } catch (e) {
        this.intake.extractError = 'Could not read your description. Please try again.';
      } finally {
        this.intake.extracting = false;
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
        guidance: null,
        guidanceLoading: false,
      };
      this.error = '';
    },
  };
}
