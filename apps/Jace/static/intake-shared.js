/**
 * intake-shared.js — logic shared between the clinician intake (app.js) and
 * the patient self-check (self-check.js). Plain global IIFE (window.VHIntake),
 * the same pattern as vh-assets/vita.js and demo_character_seeds.js — this
 * repo has no build step and no ES modules, so a shared module is a global,
 * not an import.
 *
 * Every function here is PURE: parameters in, a value out, no `this`, no
 * fetch, no clinical fact computed (same invariant app.js's own header
 * comment states — this file only reshapes/formats what the API already
 * returned). That purity is what makes it safe to share between a clinician
 * page and a patient page: neither caller's state leaks into the other's.
 *
 * Must load before app.js / self-check.js, and before the Alpine CDN tag.
 */
(function () {
  'use strict';

  // Fresh three-stage intake state (input -> confirm -> result) — the same shape for both
  // pages, so the confirm-screen markup in self-check.html can bind to `intake.fields`,
  // `intake.extraction`, `intake.hoverSpan` etc. exactly like index.html does, rather than a
  // page-specific structure. Callers extend the returned object with their own extra fields
  // (app.js adds nothing; self-check.js adds `guidance`/`describeHelp` state for its two
  // patient-only GenAI panels) rather than this factory trying to anticipate every caller.
  function makeIntake() {
    return {
      stage: 'input',            // 'input' | 'confirm' | 'result'
      note: '',
      vitals: { hr: '', sbp: '', dbp: '', rr: '', o2: '', device: '', temp: '', temp_unit: 'C' },
      noVitals: false,
      extracting: false,
      extractError: '',
      extraction: null,          // raw /api/extract (or /api/self-check/extract) response
      fields: null,               // editable confirm-screen copies (see buildConfirmFields)
      flagsAck: false,            // one acknowledgement for the guardrail-notice list
      hoverSpan: null,
      predicting: false,
      predictError: '',
      refusal: null,              // refusal_reason when the model refused
      result: null,                // /api/predict (or /api/self-check) response
    };
  }

  function escapeHtml(s) {
    return String(s || '')
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;')
      .replaceAll("'", '&#039;');
  }

  // followUpOffset (optional; self-check only — app.js never passes it, so a clinician note
  // always takes the single-segment path unchanged): the index in `note` where the patient's
  // §Describe-help follow-up answer begins (api.py's /api/self-check/extract
  // `follow_up_offset`). When present, everything from that point on is wrapped in <em> so the
  // confirm screen shows which part of the note was added afterward — hoverSpan highlighting
  // still works independently within each segment.
  function _markSpan(segment, hoverSpan) {
    if (!hoverSpan) return escapeHtml(segment);
    const idx = segment.toLowerCase().indexOf(String(hoverSpan).toLowerCase());
    if (idx < 0) return escapeHtml(segment);
    const before = segment.slice(0, idx);
    const hit = segment.slice(idx, idx + String(hoverSpan).length);
    const after = segment.slice(idx + String(hoverSpan).length);
    return `${escapeHtml(before)}<mark>${escapeHtml(hit)}</mark>${escapeHtml(after)}`;
  }

  function highlightedNote(note, hoverSpan, followUpOffset) {
    note = note || '';
    const hasBoundary = Number.isInteger(followUpOffset) && followUpOffset >= 0 && followUpOffset <= note.length;
    if (!hasBoundary) return _markSpan(note, hoverSpan);
    const head = note.slice(0, followUpOffset);
    const tail = note.slice(followUpOffset);
    return `${_markSpan(head, hoverSpan)}<em>${_markSpan(tail, hoverSpan)}</em>`;
  }

  function num(v) {
    if (v === '' || v === null || v === undefined) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function sourceMeta(source) {
    const s = String(source || '').toLowerCase();
    if (s.includes('live')) return { label: 'LIVE_GENAI', dot: 'dot--live' };
    if (s.includes('fallback')) return { label: 'FALLBACK', dot: 'dot--fallback' };
    if (s.includes('guard')) return { label: 'GUARDRAIL', dot: 'dot--guard' };
    return { label: source || 'source unknown', dot: '' };
  }

  // register: 'clinician' (default) or 'patient' — same underlying guardrail
  // flags (ctrse_core.py's extraction_guardrails), different register. The
  // clinician wording assumes a reader who knows what "paediatric" and
  // "prompt-injection" mean; the patient wording doesn't.
  function friendlyFlag(f, register) {
    const s = String(f || '');
    if (register === 'patient') {
      if (s.startsWith('injection')) {
        return "Part of what you wrote looked like an instruction rather than a symptom description, so we didn't use that part.";
      }
      if (s.startsWith('multiple')) {
        return 'Your description seems to mention more than one person. Please describe only your own symptoms.';
      }
      if (s.startsWith('paediatric')) {
        return 'This check is only set up for adults. Please seek care directly for a child.';
      }
      return s.replaceAll('_', ' ');
    }
    if (s.startsWith('injection')) return 'Possible prompt-injection wording was detected and ignored.';
    if (s.startsWith('multiple')) return 'Multiple-patient wording may be present; review before continuing.';
    if (s.startsWith('paediatric')) return 'Paediatric presentation detected; this adult workflow should not continue.';
    return s.replaceAll('_', ' ');
  }

  // extraction is the flat /api/extract (or /api/self-check/extract) response
  // — {age, sex, arrival_mode, complaints, allergies, ...} at the top level.
  // There is no `.fields` sub-object; reading through one here silently
  // blanks age/sex/arrival_mode regardless of what was actually extracted.
  function buildConfirmFields(extraction) {
    const arrivalMode = extraction?.arrival_mode || {};
    const complaints = extraction?.complaints || [];
    const allergies = extraction?.allergies || [];

    return {
      age: { value: extraction?.age?.value ?? '', span: extraction?.age?.span ?? null },
      sex: { value: extraction?.sex?.value ?? '', span: extraction?.sex?.span ?? null },
      arrival: {
        value: arrivalMode.value ?? '',
        span: arrivalMode.span ?? null,
        flagged: Boolean(arrivalMode.ambiguous),
        reason: arrivalMode.reason || '',
        alternates: arrivalMode.alternates || [],
        ack: !arrivalMode.ambiguous,
      },
      complaints: complaints.map((c) => ({
        token: c.token || c.value || '',
        span: c.span || null,
        evidence: c.evidence || null,
        fallback: Boolean(c.fallback),
        flagged: Boolean(c.ambiguous),
        alternates: c.alternates || [],
        ack: !c.ambiguous,
      })),
      allergies: allergies.length
        ? allergies.map((a) => ({ value: a.value || a.text || '', span: a.span || null }))
        : [{ value: '', span: null }],
    };
  }

  // flagsAck is the one acknowledgement checkbox for the guardrail-notice
  // list as a whole — separate from fields/extraction, so it's a third
  // parameter rather than something derivable from the other two.
  function allAcked(fields, extraction, flagsAck) {
    if (!fields) return false;
    if ((extraction?.guardrail_flags || []).length && !flagsAck) return false;
    const arr = fields.arrival;
    if (arr?.flagged && !arr.ack) return false;
    return (fields.complaints || []).every((c) => !c.flagged || c.ack);
  }

  function vitalsSummary(vitals, noVitals) {
    if (noVitals) return 'No vitals recorded at triage';
    const v = vitals || {};
    const parts = [];
    if (v.hr) parts.push(`HR ${v.hr}`);
    if (v.sbp || v.dbp) parts.push(`BP ${v.sbp || '—'}/${v.dbp || '—'}`);
    if (v.o2) parts.push(`SpO₂ ${v.o2}${v.device ? ` ${v.device}` : ''}`);
    if (v.rr) parts.push(`RR ${v.rr}`);
    if (v.temp) parts.push(`Temp ${v.temp}°${v.temp_unit || 'C'}`);
    return parts.join(' · ') || 'No vitals entered';
  }

  window.VHIntake = {
    makeIntake,
    escapeHtml,
    highlightedNote,
    num,
    sourceMeta,
    friendlyFlag,
    buildConfirmFields,
    allAcked,
    vitalsSummary,
  };
})();
