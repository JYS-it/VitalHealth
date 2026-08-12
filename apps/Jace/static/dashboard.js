/* VitalHealth portal dashboard — Alpine component.
 *
 * Two audiences, one component: a patient sees their own progress through the
 * three modules, a clinician sees every patient. Which one you get is decided
 * by the server (dashboard_api.py reads the signed session cookie); the branch
 * below only picks a layout. Editing `role` in devtools changes nothing about
 * what the API will hand back.
 *
 * Like app.js, this file renders supplied facts and derives no clinical fact of
 * its own — every headline, tone and label is formatted server-side in
 * dashboard_summaries.py.
 */

document.addEventListener('alpine:init', () => {
  Alpine.data('vhDashboard', () => ({
    loading: true,
    error: '',
    role: null,
    displayName: '',

    // patient view
    summary: null,

    // clinician view
    patients: [],
    unassigned: [],
    search: '',
    selected: null,        // loaded patient detail
    loadingDetail: false,
    activeSubject: null,   // patient the next assessment will be filed under

    get isClinician() {
      return this.role === 'clinician';
    },

    async load() {
      this.loading = true;
      this.error = '';
      try {
        const me = await (await fetch('api/dashboard/me')).json();
        if (!me.authenticated) {
          this.error = 'Your session has expired. Please log in again.';
          return;
        }
        this.role = me.role;
        this.displayName = me.display_name || '';
        await (this.isClinician ? this.loadPatients() : this.loadSummary());
      } catch (e) {
        this.error = 'Could not load your dashboard. Please refresh the page.';
      } finally {
        this.loading = false;
      }
    },

    async loadSummary() {
      const res = await fetch('api/dashboard/summary');
      if (!res.ok) {
        this.error = (await res.json().catch(() => ({}))).detail
          || 'Saved records could not be read right now.';
        return;
      }
      this.summary = await res.json();
    },

    async loadPatients() {
      const query = this.search.trim() ? `?q=${encodeURIComponent(this.search.trim())}` : '';
      const res = await fetch(`api/dashboard/patients${query}`);
      if (!res.ok) {
        this.error = (await res.json().catch(() => ({}))).detail
          || 'Saved records could not be read right now.';
        return;
      }
      const data = await res.json();
      this.patients = data.patients || [];
      this.unassigned = data.unassigned || [];
      this.activeSubject = data.active_subject || null;
    },

    async openPatient(patientId) {
      this.loadingDetail = true;
      try {
        const res = await fetch(`api/dashboard/patients/${patientId}`);
        this.selected = res.ok ? await res.json() : null;
      } catch (e) {
        this.selected = null;
      } finally {
        this.loadingDetail = false;
      }
    },

    closePatient() {
      this.selected = null;
    },

    /* Neither the triage nor the stroke form has a patient field, so a
     * clinician has to say who they are working on before running one. The
     * gateway stores it in a signed cookie all three apps can read. */
    async setSubject(patient) {
      const clearing = !patient || this.activeSubject === patient.external_id;
      const res = await fetch('/api/clinician/context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_external_id: clearing ? null : patient.external_id,
          display_name: clearing ? null : patient.display_name,
        }),
      });
      if (res.ok) {
        this.activeSubject = clearing ? null : patient.external_id;
      }
    },

    activeSubjectName() {
      const match = this.patients.find((p) => p.external_id === this.activeSubject);
      return match ? match.display_name : this.activeSubject;
    },

    moduleList(patient) {
      return ['triage', 'stroke', 'emc'].map((key) => patient.modules[key]);
    },

    when(iso) {
      if (!iso) return '';
      const parsed = new Date(iso);
      if (Number.isNaN(parsed.getTime())) return '';
      return parsed.toLocaleString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit',
      });
    },
  }));
});
