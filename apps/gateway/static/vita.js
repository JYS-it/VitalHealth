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
      document.addEventListener('DOMContentLoaded', buildWidget, { once: true });
      return;
    }
    buildWidget();
  }

  init();
})();
