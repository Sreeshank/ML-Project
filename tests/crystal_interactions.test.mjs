// Exercise the real renderer with a lightweight DOM/canvas harness, without a browser dependency.
// Run: node --test tests/crystal_interactions.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

const source = await readFile(new URL('../assets/crystal.js', import.meta.url), 'utf8');
const { default: mount } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

class Element extends EventTarget {
  constructor(dataset = {}) { super(); this.dataset = dataset; this.attributes = {}; this.textContent = ''; }
  setAttribute(name, value) { this.attributes[name] = value; }
  fire(type, props = {}) { const event = new Event(type, { cancelable: true }); Object.assign(event, props); this.dispatchEvent(event); return event; }
}

function harness({ paused = false, reduced = false } = {}) {
  const pending = new Map(); let id = 0; const observers = []; const lines = [];
  globalThis.requestAnimationFrame = fn => { pending.set(++id, fn); return id; };
  globalThis.cancelAnimationFrame = key => pending.delete(key);
  const motion = new Element(); motion.matches = reduced;
  globalThis.window = { matchMedia: () => motion, devicePixelRatio: 3 };
  globalThis.document = new Element(); document.hidden = false;
  class Observer {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe() {}
    disconnect() { this.disconnected = true; }
  }
  globalThis.ResizeObserver = Observer;
  globalThis.IntersectionObserver = Observer;
  const canvas = new Element();
  const ctx = new Proxy({}, { get: (_, key) => {
    if (key === 'createRadialGradient') return () => ({ addColorStop() {} });
    if (key === 'lineTo') return (x, y) => lines.push([x, y]);
    return () => {};
  }, set: () => true });
  canvas.getContext = () => ctx;
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 460, height: 238 });
  canvas.setPointerCapture = () => {};
  canvas.focus = () => {};
  const detail = new Element(), zoom = new Element();
  const sites = ['A', 'B', 'X'].map(site => new Element({ site }));
  const actions = Object.fromEntries(['rotate', 'bonds', 'out', 'in', 'reset'].map(action => [action, new Element({ action })]));
  const root = {
    querySelector(selector) {
      if (selector === 'canvas') return canvas;
      if (selector === '.site-detail') return detail;
      if (selector === '[data-zoom]') return zoom;
      return actions[selector.match(/data-action="(.+)"/)[1]];
    },
    querySelectorAll: selector => selector === '[data-site]' ? sites : Object.values(actions),
  };
  const cleanup = mount({ parentElement: { querySelector: () => root }, data: { paused } });
  function frame(time = 1000) {
    const jobs = [...pending.values()]; pending.clear();
    for (const fn of jobs) fn(time);
  }
  return { canvas, detail, zoom, sites, actions, motion, frame, cleanup, pending, observers, lines };
}

test('zoom clamps, atom inspection, bond toggle, and reset work independently of Python', () => {
  const ui = harness();
  try {
    ui.frame();
    assert.equal(ui.canvas.width, 920, 'Pixel ratio is capped at 2');
    for (let i = 0; i < 20; i++) ui.actions.in.fire('click');
    assert.equal(ui.zoom.textContent, '145%');
    for (let i = 0; i < 30; i++) ui.actions.out.fire('click');
    assert.equal(ui.zoom.textContent, '65%');
    ui.sites[1].fire('click');
    assert.match(ui.detail.textContent, /B site.*Body center/);
    assert.equal(ui.sites[1].attributes['aria-pressed'], 'true');
    ui.actions.bonds.fire('click');
    assert.equal(ui.actions.bonds.attributes['aria-pressed'], 'false');
    ui.actions.reset.fire('click');
    assert.equal(ui.zoom.textContent, '100%');
    assert.ok(ui.sites.every(site => site.attributes['aria-pressed'] === 'false'));
  } finally { ui.cleanup(); }
});

test('drag and keyboard change the view, pause spin, and preserve page navigation keys', () => {
  const ui = harness();
  try {
    ui.frame(); const initial = ui.lines.slice(-20); ui.lines.length = 0;
    ui.canvas.fire('pointerdown', { button: 0, pointerId: 1, clientX: 220, clientY: 110 });
    ui.canvas.fire('pointermove', { clientX: 300, clientY: 140 });
    ui.canvas.fire('pointerup'); ui.frame();
    assert.notDeepEqual(ui.lines.slice(-20), initial, 'Rotation changes the projected geometry');
    assert.equal(ui.actions.rotate.attributes['aria-pressed'], 'false');
    assert.equal(ui.canvas.fire('keydown', { key: 'ArrowLeft' }).defaultPrevented, true);
    assert.equal(ui.canvas.fire('keydown', { key: 'Tab' }).defaultPrevented, false);
    ui.canvas.fire('keydown', { key: '+' });
    assert.equal(ui.zoom.textContent, '110%');
    ui.canvas.fire('keydown', { key: 'Home' });
    assert.equal(ui.zoom.textContent, '100%');
  } finally { ui.cleanup(); }
});

test('reduced motion and global pause stop continuous frames; cleanup removes resources', () => {
  for (const options of [{ reduced: true }, { paused: true }]) {
    const ui = harness(options);
    ui.frame();
    assert.equal(ui.pending.size, 0);
    assert.equal(ui.actions.rotate.attributes['aria-pressed'], 'false');
    if (options.paused) {
      assert.equal(ui.actions.rotate.disabled, true);
      assert.equal(ui.actions.rotate.textContent, 'Motion paused');
      ui.canvas.fire('keydown', { key: ' ' }); ui.frame();
      assert.equal(ui.pending.size, 0);
    }
    ui.actions.in.fire('click'); ui.frame();
    assert.equal(ui.zoom.textContent, '110%', 'Manual controls remain usable');
    ui.cleanup();
    assert.equal(ui.pending.size, 0);
    assert.ok(ui.observers.every(observer => observer.disconnected));
    ui.actions.in.fire('click');
    assert.equal(ui.zoom.textContent, '110%', 'Listeners were removed on unmount');
  }
});

test('off-screen and background views stop their animation loop', () => {
  const ui = harness();
  try {
    ui.frame(); assert.equal(ui.pending.size, 1);
    ui.observers[1].callback([{ isIntersecting: false }]); ui.frame();
    assert.equal(ui.pending.size, 0);
    ui.observers[1].callback([{ isIntersecting: true }]); ui.frame();
    assert.equal(ui.pending.size, 1);
    document.hidden = true; document.fire('visibilitychange'); ui.frame();
    assert.equal(ui.pending.size, 0);
    document.hidden = false; document.fire('visibilitychange');
    assert.equal(ui.pending.size, 1);
  } finally { ui.cleanup(); }
});
