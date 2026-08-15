/**
 * self-check.js — the patient triage self-check page's own, small Alpine
 * root. Deliberately separate from app.js: this page renders only what
 * /api/self-check/options and /api/self-check return, computes no clinical
 * fact of its own (predict_from_fields, patient_urgency_band and
 * patient_view all live in ctrse_core.py — see its §Patient self-check
 * section), and never calls a clinician-only route.
 */
function selfCheckApp() {
  return {
    stage: 'intro',           // 'intro' | 'form' | 'result'
    error: '',
    submitting: false,

    emergencyNumber: '995',
    crisisNumber: '',
    crisisLabel: '',
    scopeNote: '',
    symptomOptions: [],

    age: null,
    sex: '',
    arrivalMode: '',
    complaints: [],
    noVitals: false,
    vitals: { hr: null, sbp: null, dbp: null, rr: null, o2: null, temp: null, temp_unit: 'C' },

    result: null,

    async init() {
      try {
        const res = await fetch('api/self-check/options');
        if (!res.ok) throw new Error('options request failed');
        const data = await res.json();
        this.symptomOptions = Array.isArray(data.complaints) ? data.complaints : [];
        const contacts = data.emergency_contacts || {};
        // Fall back to the same numbers the banner markup defaults to, so a
        // failed fetch never leaves the banner blank.
        this.emergencyNumber = contacts.emergency_number || this.emergencyNumber;
        this.crisisNumber = contacts.crisis_number || '';
        this.crisisLabel = contacts.crisis_label || '';
        this.scopeNote = data.scope_note || '';
      } catch (e) {
        // The emergency banner still renders with its default numbers even
        // if this fetch fails — it must never depend on a network call to
        // say something is wrong.
        this.error = 'Some information on this page could not load. The emergency numbers above are still correct.';
      }
    },

    toggleSymptom(token, checked) {
      if (checked) {
        if (!this.complaints.includes(token)) this.complaints.push(token);
      } else {
        this.complaints = this.complaints.filter((t) => t !== token);
      }
    },

    bandColour(tone) {
      return { critical: 'var(--p1)', warning: 'var(--warn)', neutral: 'var(--accent)' }[tone] || 'var(--accent)';
    },

    async submit() {
      this.error = '';
      this.submitting = true;
      window.VitalHealthLoading?.show('Checking your symptoms');
      try {
        const vitals = this.noVitals
          ? {}
          : Object.fromEntries(
              Object.entries(this.vitals).filter(([, v]) => v !== null && v !== '')
            );
        const body = {
          age: this.age,
          sex: this.sex || null,
          arrival_mode: this.arrivalMode || null,
          complaints: this.complaints.map((token) => ({ token })),
          vitals,
        };
        const res = await fetch('api/self-check', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail || 'Something went wrong submitting your check. Please try again.');
        }
        this.result = await res.json();
        this.stage = 'result';
      } catch (e) {
        this.error = e.message || 'Something went wrong submitting your check. Please try again.';
      } finally {
        this.submitting = false;
        window.VitalHealthLoading?.hide();
      }
    },

    reset() {
      this.result = null;
      this.age = null;
      this.sex = '';
      this.arrivalMode = '';
      this.complaints = [];
      this.noVitals = false;
      this.vitals = { hr: null, sbp: null, dbp: null, rr: null, o2: null, temp: null, temp_unit: 'C' };
      this.error = '';
      this.stage = 'intro';
    },
  };
}
