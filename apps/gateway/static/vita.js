(function () {
  const FALLBACK = 'Vita is currently unavailable. Please try again later. VitalHealth outputs are educational decision-support only and should be reviewed by a qualified healthcare professional.';
  const QUICK_PROMPTS = [
    'How do I use Stroke Assessment?',
    'How do I use Clinical Triage?',
    'How do I use the EMC Workflow?',
    'Where will saved workflows appear?',
    'Is this a diagnosis?',
  ];

  function getSectionContext() {
    const bodySection = document.body && document.body.dataset && document.body.dataset.vitalhealthSection;
    if (bodySection) return bodySection;

    const path = window.location.pathname.toLowerCase();
    if (path.startsWith('/stroke')) return 'stroke';
    if (path.startsWith('/emc')) return 'emc';
    if (path.startsWith('/triage')) return 'clinical triage';
    return 'dashboard';
  }

  function ensureStylesheet(href, id) {
    if (document.getElementById(id)) return;
    const link = document.createElement('link');
    link.id = id;
    link.rel = 'stylesheet';
    link.href = href;
    document.head.appendChild(link);
  }

  function activeSection() {
    const path = window.location.pathname.toLowerCase();
    if (path.startsWith('/stroke')) return 'stroke';
    if (path.startsWith('/emc')) return 'emc';
    if (path.includes('self-check')) return 'triage';
    return 'home';
  }

  function escapeMarkup(value) {
    return String(value).replace(/[&<>"']/g, (character) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[character]);
  }

  function navMarkup(role) {
    const triageHref = role === 'patient' ? '/triage/self-check.html' : '/triage/';
    const triageLabel = role === 'patient' ? 'Triage Self-Check' : 'Clinical Triage';
    const active = activeSection();
    const link = (key, href, label) =>
      `<a class="${active === key ? 'active' : ''}" href="${href}">${label}</a>`;
    return [
      link('home', '/triage/', 'Home'),
      link('triage', triageHref, triageLabel),
      link('stroke', '/stroke/', 'Stroke Assessment'),
      link('emc', '/emc/', 'EMC Workflow'),
    ].join('');
  }

  async function buildSharedHeader() {
    const header = document.querySelector('.site-header, .header, .topbar');
    if (!header) return;

    let me;
    try {
      const response = await fetch('/api/me', { credentials: 'same-origin' });
      if (!response.ok) return;
      me = await response.json();
    } catch (error) {
      return;
    }
    if (!me || !me.authenticated) return;

    const displayName = me.display_name || me.email || 'User';
    let account = header.querySelector('.auth-status');
    if (!account) {
      account = document.createElement('div');
      account.className = 'auth-status';
      const topRow = header.querySelector('.header__top, .header-wrap, .topbar-inner');
      if (topRow) topRow.appendChild(account);
    }
    if (account) {
      const initial = displayName.trim().charAt(0).toUpperCase() || 'U';
      account.innerHTML = [
        `<span class="auth-status__avatar" aria-hidden="true">${initial}</span>`,
        '<span class="auth-status__copy">',
        '  <span class="auth-status__label">Signed in as</span>',
        `  <strong class="auth-status__user">${escapeMarkup(displayName)}</strong>`,
        '</span>',
        '<a class="auth-status__logout" href="/logout">Log out</a>',
      ].join('');
    }

    let nav = header.querySelector('.global-nav, .navbar');
    if (!nav) {
      const navWrap = document.createElement('div');
      navWrap.className = 'container nav-wrap';
      nav = document.createElement('nav');
      nav.className = 'navbar global-nav';
      nav.setAttribute('aria-label', 'VitalHealth navigation');
      navWrap.appendChild(nav);
      header.appendChild(navWrap);
    }

    // Link-based workflow headers can share one role-aware navigation. The
    // dashboard keeps its Alpine buttons because they switch in-page views.
    if (!nav.querySelector('button')) nav.innerHTML = navMarkup(me.role);
  }

  function loadingLabelFor(text) {
    const label = String(text || '').trim().toLowerCase();
    if (label.includes('care plan')) return 'Generating care plan';
    if (label.includes('extract')) return 'Extracting clinical details';
    if (label.includes('predict') || label.includes('check my symptoms')) return 'Calculating assessment output';
    if (label.includes('regenerate')) return 'Regenerating draft';
    if (label.includes('approve')) return 'Approving and preparing the result';
    if (label.includes('reject')) return 'Processing rejection';
    if (label.includes('submit')) return 'Submitting your request';
    return 'Generating your output';
  }

  function ensureLoadingOverlay() {
    let overlay = document.getElementById('loading-overlay');
    if (overlay) return overlay;
    overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay vh-loading-overlay';
    overlay.hidden = true;
    overlay.setAttribute('aria-live', 'polite');
    overlay.setAttribute('aria-busy', 'true');
    overlay.innerHTML = [
      '<div class="loading-dialog vh-loading-dialog" role="status">',
      '  <span class="loading-label">Generating your output</span>',
      '  <div class="skeleton title"></div>',
      '  <div class="skeleton line"></div>',
      '  <div class="skeleton line"></div>',
      '  <div class="skeleton line short"></div>',
      '</div>',
    ].join('');
    document.body.appendChild(overlay);
    return overlay;
  }

  function showLoading(label) {
    const overlay = ensureLoadingOverlay();
    const labelNode = overlay.querySelector('.loading-label');
    if (labelNode) labelNode.textContent = label || 'Generating your output';
    overlay.hidden = false;
    document.body.classList.add('vh-is-loading');
  }

  function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.hidden = true;
    document.body.classList.remove('vh-is-loading');
  }

  function installLoadingStates() {
    window.VitalHealthLoading = { show: showLoading, hide: hideLoading };

    document.addEventListener('submit', (event) => {
      const form = event.target;
      if (!(form instanceof HTMLFormElement) || !form.checkValidity()) return;
      if ((form.method || 'get').toLowerCase() !== 'post') return;
      const submitter = event.submitter;
      const text = submitter && (submitter.textContent || submitter.value);
      showLoading(loadingLabelFor(text));
    }, true);

    document.addEventListener('click', (event) => {
      if (!(event.target instanceof Element)) return;
      const link = event.target.closest('a[data-loading-label], a[href*="/care-plan"]');
      if (!link) return;
      showLoading(link.dataset.loadingLabel || loadingLabelFor(link.textContent));
    });
  }

  function buildWidget() {
    if (document.getElementById('vitalhealth-vita-root')) return;

    ensureStylesheet('/vh-assets/vita.css', 'vitalhealth-vita-css');

    const root = document.createElement('section');
    root.id = 'vitalhealth-vita-root';
    root.className = 'vita-shell';
    root.setAttribute('aria-label', 'Vita assistant');

    root.innerHTML = [
      '<button type="button" class="vita-fab" aria-label="Open Vita assistant" aria-expanded="false" aria-controls="vitalhealth-vita-panel">',
      '  <span class="vita-face" aria-hidden="true">',
      '    <span class="vita-eyes"><i></i><i></i></span>',
      '    <span class="vita-mouth"></span>',
      '  </span>',
      '</button>',
      '<div class="vita-panel card" id="vitalhealth-vita-panel" hidden>',
      '  <header class="vita-head">',
      '    <div>',
      '      <h3>Vita</h3>',
      '      <p>VitalHealth assistant</p>',
      '    </div>',
      '    <button type="button" class="btn-ghost vita-minimize" data-vita-close="1" aria-label="Minimize Vita assistant">Minimize</button>',
      '  </header>',
      '  <div class="vita-quick" role="group" aria-label="Quick prompts"></div>',
      '  <div class="vita-messages" aria-live="polite"></div>',
      '  <form class="vita-composer">',
      '    <input type="text" maxlength="500" aria-label="Ask Vita" placeholder="Ask Vita about navigation or workflow...">',
      '    <button type="submit" class="btn-generate vita-send">Send</button>',
      '  </form>',
      '</div>',
    ].join('');

    const fab = root.querySelector('.vita-fab');
    const panel = root.querySelector('.vita-panel');
    const minimize = root.querySelector('.vita-minimize');
    const quickWrap = root.querySelector('.vita-quick');
    const messages = root.querySelector('.vita-messages');
    const form = root.querySelector('.vita-composer');
    const input = form.querySelector('input');
    const send = form.querySelector('.vita-send');
    const mouth = root.querySelector('.vita-mouth');
    const head = root.querySelector('.vita-head');

    let open = false;
    let loading = false;
    let drag = null;
    let dragDX = 0;
    let dragDY = 0;

    function applyPosition() {
      root.style.transform = 'translate(' + dragDX + 'px, ' + dragDY + 'px)';
    }

    function safeClosePanel() {
      if (!open) return;
      fab.focus();
      setOpen(false);
    }

    function scrollBottom() {
      messages.scrollTop = messages.scrollHeight;
    }

    function setOpen(next) {
      open = next;
      panel.hidden = !open;
      panel.style.display = open ? 'grid' : 'none';
      panel.setAttribute('aria-hidden', String(!open));
      fab.setAttribute('aria-expanded', String(open));
      root.classList.toggle('vita-shell--open', open);
      root.classList.toggle('vita-shell--opening', false);
      if (open) {
        window.setTimeout(scrollBottom, 0);
        window.setTimeout(() => input.focus(), 0);
        mouth.style.transform = 'scaleY(1.1)';
      } else {
        mouth.style.transform = 'scaleY(1)';
      }
    }

    function setLoading(next) {
      loading = next;
      send.disabled = loading || !input.value.trim();
      input.disabled = loading;
    }

    function addMessage(role, text) {
      const article = document.createElement('article');
      article.className = 'vita-msg ' + (role === 'user' ? 'vita-msg--user' : 'vita-msg--bot');
      const p = document.createElement('p');
      p.textContent = text;
      article.appendChild(p);
      messages.appendChild(article);
      scrollBottom();
    }

    function renderQuickPrompts() {
      QUICK_PROMPTS.forEach((prompt) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'vita-chip';
        btn.textContent = prompt;
        btn.addEventListener('click', () => {
          input.value = prompt;
          submitPrompt();
        });
        quickWrap.appendChild(btn);
      });
    }

    async function submitPrompt() {
      const message = input.value.trim();
      if (!message || loading) return;
      addMessage('user', message);
      input.value = '';
      setLoading(true);
      scrollBottom();

      try {
        const response = await fetch('/api/vita/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message,
            context: { active_section: getSectionContext() },
          }),
        });
        const data = await response.json();
        const reply = data && data.success && data.data && typeof data.data.reply === 'string'
          ? data.data.reply
          : FALLBACK;
        addMessage('bot', reply);
      } catch (error) {
        addMessage('bot', FALLBACK);
      } finally {
        setLoading(false);
      }
    }

    fab.addEventListener('click', () => {
      if (open) {
        setOpen(false);
        return;
      }
      root.classList.add('vita-shell--opening');
      window.setTimeout(() => {
        root.classList.remove('vita-shell--opening');
        setOpen(true);
      }, 140);
    });

    minimize.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      safeClosePanel();
    });

    function beginDrag(event) {
      if (event.button !== 0) return;
      if (event.target instanceof Element) {
        if (event.target.closest('button, input, textarea, select, a, label')) return;
      }

      const nextDrag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        startDX: dragDX,
        startDY: dragDY,
        moved: false,
      };
      drag = nextDrag;
    }

    function moveDrag(event) {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const deltaX = event.clientX - drag.startX;
      const deltaY = event.clientY - drag.startY;

      if (!drag.moved) {
        if (Math.abs(deltaX) < 4 && Math.abs(deltaY) < 4) {
          return;
        }
        drag.moved = true;
        root.classList.add('vita-shell--dragging');
      }

      // Keep widget within viewport bounds while dragging.
      const maxX = Math.max(0, window.innerWidth - 76);
      const maxY = Math.max(0, window.innerHeight - 76);
      dragDX = Math.max(-maxX, Math.min(maxX, drag.startDX + deltaX));
      dragDY = Math.max(-maxY, Math.min(maxY, drag.startDY + deltaY));
      applyPosition();
      event.preventDefault();
    }

    function endDrag(event) {
      if (!drag || drag.pointerId !== event.pointerId) return;
      const moved = drag.moved;
      drag = null;
      root.classList.remove('vita-shell--dragging');
      if (moved) {
        event.preventDefault();
        event.stopPropagation();
      }
    }

    fab.addEventListener('pointerdown', beginDrag);
    head.addEventListener('pointerdown', beginDrag);
    document.addEventListener('pointermove', moveDrag);
    document.addEventListener('pointerup', endDrag);
    document.addEventListener('pointercancel', endDrag);

    root.addEventListener('click', (event) => {
      if (!(event.target instanceof Element)) return;
      if (event.target.closest('[data-vita-close="1"]')) {
        event.preventDefault();
        event.stopPropagation();
        safeClosePanel();
      }
    });

    document.addEventListener('pointerdown', (event) => {
      if (!open) return;
      if (!(event.target instanceof Node)) return;
      if (!root.contains(event.target)) {
        safeClosePanel();
      }
    });

    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        safeClosePanel();
      }
    });

    form.addEventListener('submit', (event) => {
      event.preventDefault();
      submitPrompt();
    });
    input.addEventListener('input', () => setLoading(loading));
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submitPrompt();
      }
    });

    renderQuickPrompts();
    addMessage('bot', 'Hi, I\'m Vita. I can guide you through VitalHealth workflows in simple steps.');
    panel.style.display = 'none';
    panel.setAttribute('aria-hidden', 'true');
    setLoading(false);
    document.body.appendChild(root);
  }

  function init() {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        buildSharedHeader();
        installLoadingStates();
        buildWidget();
      }, { once: true });
      return;
    }
    buildSharedHeader();
    installLoadingStates();
    buildWidget();
  }

  init();
})();
