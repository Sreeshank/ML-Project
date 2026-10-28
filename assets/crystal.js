// Original local canvas renderer. No network calls, WebGL libraries, or Python reruns.
export default function ({ parentElement, data }) {
  const root = parentElement.querySelector('.viewer');
  const canvas = root.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const detail = root.querySelector('.site-detail');
  if (!ctx) { detail.textContent = 'Canvas is unavailable. Idealized ABX₃: A at cube corners, B at the center, X at face centers.'; return; }
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const abort = new AbortController();
  const listen = (el, type, fn, options = {}) => el.addEventListener(type, fn, { ...options, signal: abort.signal });
  const colors = { A: '#65e9da', B: '#c6a0ff', X: '#ffc58a' };
  const descriptions = {
    A: 'A site · Cube corners. Often an organic or inorganic cation; e.g. MA, FA, or Cs.',
    B: 'B site · Body center. A metal cation; e.g. Pb or Sn, coordinated by six X sites.',
    X: 'X site · Face centers. Halide anions form the octahedron around B; e.g. I, Br, or Cl.',
  };
  const atoms = [];
  for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) atoms.push({ x, y, z, site: 'A' });
  atoms.push({ x: 0, y: 0, z: 0, site: 'B' });
  for (let axis = 0; axis < 3; axis++) for (const sign of [-1, 1]) {
    const xyz = [0, 0, 0]; xyz[axis] = sign;
    atoms.push({ x: xyz[0], y: xyz[1], z: xyz[2], site: 'X' });
  }
  let rx = -.33, ry = .62, zoom = 1, width = 450, height = 238;
  let auto = !motion.matches && !data?.paused, bonds = true, selected = null, hover = null;
  let dragging = false, pointer = null, last = null, frame = 0, lastTime = 0, phase = 0;
  let visible = true, disposed = false, projected = [];
  let tiltX = 0, tiltY = 0;
  const rotateButton = root.querySelector('[data-action="rotate"]');
  const bondsButton = root.querySelector('[data-action="bonds"]');
  function syncControls() {
    rotateButton.textContent = data?.paused ? 'Motion paused' : auto ? 'Ⅱ Pause spin' : '▷ Auto spin';
    rotateButton.setAttribute('aria-pressed', String(auto));
    bondsButton.setAttribute('aria-pressed', String(bonds));
    root.querySelector('[data-zoom]').textContent = `${Math.round(zoom * 100)}%`;
    root.querySelectorAll('[data-site]').forEach(button => button.setAttribute('aria-pressed', String(selected === button.dataset.site)));
  }
  function inspect(site) {
    const text = descriptions[site] || 'Explore an idealized unit cell. This illustration does not simulate your inputs.';
    if (detail.textContent !== text) detail.textContent = text;
  }
  function project({ x, y, z }) {
    const cy = Math.cos(ry + tiltY), sy = Math.sin(ry + tiltY);
    const cx = Math.cos(rx + tiltX), sx = Math.sin(rx + tiltX);
    const px = x * cy + z * sy, zz = -x * sy + z * cy;
    const py = y * cx - zz * sx, pz = y * sx + zz * cx;
    const depth = 5 / (5 + pz), scale = Math.min(width * .19, height * .25) * zoom;
    return { x: width / 2 + px * scale * depth, y: height / 2 + py * scale * depth - 7, z: pz, depth };
  }
  function line(a, b, color, lineWidth = 1) {
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.stroke();
  }
  function draw() {
    ctx.clearRect(0, 0, width, height);
    // Subtle reticle and perspective floor establish depth without implying measured axes.
    ctx.strokeStyle = '#7396be0c'; ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 25) line({ x, y: 0 }, { x, y: height }, '#7396be0c');
    for (let y = 0; y < height; y += 25) line({ x: 0, y }, { x: width, y }, '#7396be0c');
    const radius = Math.min(width * .37, height * .44);
    ctx.beginPath(); ctx.ellipse(width / 2, height / 2 - 7, radius * 1.35, radius * .6, -.28, 0, Math.PI * 2);
    ctx.strokeStyle = '#65e9da22'; ctx.stroke();
    ctx.beginPath(); ctx.arc(width / 2, height / 2 - 7, radius, 0, Math.PI * 2);
    ctx.setLineDash([2, 9]); ctx.strokeStyle = '#9bafff28'; ctx.stroke(); ctx.setLineDash([]);
    for (let i = 0; i < 36; i++) {
      const a = i * 2.39996 + phase * .11, r = radius * (1.13 + .14 * Math.sin(i * 8.3));
      let x = width / 2 + Math.cos(a) * r * 1.25, y = height / 2 - 7 + Math.sin(a) * r * .76;
      if (pointer && !motion.matches && !data?.paused) {
        const dx = x - pointer.x, dy = y - pointer.y, dist = Math.hypot(dx, dy);
        if (dist > 0 && dist < 65) { x += dx / dist * (65 - dist) * .3; y += dy / dist * (65 - dist) * .3; }
      }
      ctx.beginPath(); ctx.arc(x, y, i % 5 === 0 ? 1.8 : .8, 0, Math.PI * 2);
      ctx.fillStyle = i % 3 === 0 ? '#c6a0ff88' : '#65e9da66'; ctx.fill();
    }
    projected = atoms.map((atom, i) => ({ ...project(atom), site: atom.site, index: i }));
    for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
      const a = atoms[i], b = atoms[j];
      if (Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z) === 2) line(projected[i], projected[j], '#65e9da55');
    }
    if (bonds) {
      for (let i = 9; i < atoms.length; i++) {
        line(projected[8], projected[i], '#c6a0ff80', 1.3);
        for (let j = i + 1; j < atoms.length; j++) {
          const a = atoms[i], b = atoms[j];
          if (a.x * b.x + a.y * b.y + a.z * b.z === 0) line(projected[i], projected[j], '#c6a0ff30');
        }
      }
    }
    [...projected].sort((a, b) => b.z - a.z).forEach(p => {
      const active = (hover ?? selected) === p.site;
      const radius = (p.site === 'B' ? 10 : p.site === 'A' ? 6.3 : 5.3) * p.depth * Math.sqrt(zoom);
      p.radius = radius;
      if (active) { ctx.beginPath(); ctx.arc(p.x, p.y, radius + 5, 0, Math.PI * 2); ctx.strokeStyle = colors[p.site]; ctx.lineWidth = .8; ctx.stroke(); }
      ctx.shadowColor = colors[p.site]; ctx.shadowBlur = active ? 22 : 9;
      const glow = ctx.createRadialGradient(p.x - radius * .3, p.y - radius * .3, 0, p.x, p.y, radius);
      glow.addColorStop(0, '#efffff'); glow.addColorStop(.35, colors[p.site]); glow.addColorStop(1, '#1c3448');
      ctx.beginPath(); ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fillStyle = glow; ctx.fill(); ctx.shadowBlur = 0;
      if (active) { ctx.fillStyle = colors[p.site]; ctx.font = '10px monospace'; ctx.fillText(p.site, p.x + radius + 7, p.y + 3); }
    });
  }
  function schedule() { if (!disposed && visible && !document.hidden && !frame) frame = requestAnimationFrame(tick); }
  function tick(now) {
    frame = 0;
    if (disposed || !visible || document.hidden) { lastTime = 0; return; }
    const delta = lastTime ? Math.min((now - lastTime) / 1000, .05) : 0;
    lastTime = now;
    if (auto && !dragging && !hover) { ry += delta * .18; phase += delta; }
    draw();
    if (auto) schedule(); else lastTime = 0;
  }
  function resize() {
    const rect = canvas.getBoundingClientRect();
    width = Math.max(rect.width, 1); height = Math.max(rect.height, 1);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); schedule();
  }
  function setZoom(value) { zoom = Math.max(.65, Math.min(1.45, value)); syncControls(); schedule(); }
  function reset() { rx = -.33; ry = .62; tiltX = tiltY = 0; zoom = 1; selected = hover = null; inspect(null); syncControls(); schedule(); }
  function position(event) { const rect = canvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
  listen(canvas, 'pointerdown', event => {
    if (event.button !== 0) return;
    dragging = true; last = position(event); hover = null;
    canvas.setPointerCapture(event.pointerId); canvas.focus({ preventScroll: true });
    auto = false; syncControls(); schedule();
  });
  listen(canvas, 'pointermove', event => {
    pointer = position(event);
    if (dragging) { ry += (pointer.x - last.x) * .009; rx += (pointer.y - last.y) * .009; last = pointer; }
    else {
      const hit = [...projected].sort((a, b) => a.z - b.z).find(p => Math.hypot(p.x - pointer.x, p.y - pointer.y) < (p.radius || 7) + 7);
      hover = hit?.site ?? null; inspect(hover ?? selected);
      if (!motion.matches && !data?.paused) { tiltX = (pointer.y / height - .5) * .08; tiltY = (pointer.x / width - .5) * .08; }
    }
    schedule();
  });
  const release = () => { dragging = false; last = null; schedule(); };
  listen(canvas, 'pointerup', release); listen(canvas, 'pointercancel', release); listen(canvas, 'lostpointercapture', release);
  listen(canvas, 'pointerleave', () => { pointer = null; hover = null; tiltX = tiltY = 0; inspect(selected); schedule(); });
  listen(canvas, 'keydown', event => {
    const keys = ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', 'Home', ' '];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    if (event.key.startsWith('Arrow')) auto = false;
    if (event.key === 'ArrowLeft') ry -= .14;
    if (event.key === 'ArrowRight') ry += .14;
    if (event.key === 'ArrowUp') rx -= .14;
    if (event.key === 'ArrowDown') rx += .14;
    if (event.key === '+' || event.key === '=') setZoom(zoom + .1);
    if (event.key === '-') setZoom(zoom - .1);
    if (event.key === 'Home') reset();
    if (event.key === ' ' && !data?.paused) auto = !auto;
    syncControls(); schedule();
  });
  root.querySelectorAll('[data-site]').forEach(button => listen(button, 'click', () => {
    selected = selected === button.dataset.site ? null : button.dataset.site; hover = null;
    inspect(selected); syncControls(); schedule();
  }));
  root.querySelectorAll('[data-action]').forEach(button => listen(button, 'click', () => {
    switch (button.dataset.action) {
      case 'rotate': if (!data?.paused) auto = !auto; break;
      case 'bonds': bonds = !bonds; break;
      case 'out': setZoom(zoom - .1); break;
      case 'in': setZoom(zoom + .1); break;
      case 'reset': reset(); break;
    }
    syncControls(); schedule();
  }));
  if (data?.paused) { rotateButton.disabled = true; rotateButton.textContent = 'Motion paused'; }
  listen(motion, 'change', () => { if (motion.matches) auto = false; syncControls(); schedule(); });
  listen(document, 'visibilitychange', () => { lastTime = 0; schedule(); });
  const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(canvas);
  const visibilityObserver = new IntersectionObserver(entries => { visible = entries[0].isIntersecting; lastTime = 0; schedule(); });
  visibilityObserver.observe(canvas);
  syncControls(); resize();
  return () => { disposed = true; cancelAnimationFrame(frame); abort.abort(); resizeObserver.disconnect(); visibilityObserver.disconnect(); };
}
