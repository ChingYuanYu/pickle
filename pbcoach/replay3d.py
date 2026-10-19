"""Interactive 3D replay of a tracked clip: court, net, posed players, rings and trails.

replay_html(replay) returns a self-contained HTML page (three.js from a CDN) built from the
replay.json the pipeline writes. Court meters map to three.js as X = x, Y = z (up), Z = -y.
"""
from __future__ import annotations

import json

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body{margin:0;height:100%;background:#0f1115;color:#e8e8e8;font:13px system-ui,sans-serif;overflow:hidden}
  #c{position:absolute;inset:0}
  .bar{position:absolute;left:10px;right:10px;bottom:10px;display:flex;gap:8px;align-items:center;
       background:rgba(15,17,21,.8);padding:8px 10px;border-radius:10px}
  .bar input[type=range]{flex:1}
  button{background:#262b34;color:#e8e8e8;border:1px solid #3a404c;border-radius:7px;padding:5px 9px;cursor:pointer}
  button.on{background:#e8590c;border-color:#e8590c}
  .views{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap}
  .legend{position:absolute;top:10px;right:10px;background:rgba(15,17,21,.8);padding:6px 10px;border-radius:10px}
  .legend div{display:flex;align-items:center;gap:6px;margin:2px 0}
  .dot{width:10px;height:10px;border-radius:50%}
  #t{min-width:52px;text-align:right;font-variant-numeric:tabular-nums}
</style></head>
<body>
<div id="c"></div>
<div class="views" id="views"></div>
<div class="legend" id="legend"></div>
<div class="bar">
  <button id="play">⏸</button>
  <input id="seek" type="range" min="0" value="0" step="1">
  <span id="t">0.0 s</span>
  <button id="speed">1×</button>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const R = __REPLAY__;
const C = R.court, W = C.width, L = C.length;
const P = (x, y, z = 0) => new THREE.Vector3(x - W / 2, z, -(y - L / 2));   // court meters -> scene

const el = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
renderer.shadowMap.enabled = true;
el.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1115);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.maxPolarAngle = Math.PI / 2 - 0.02;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334433, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-6, 14, 8); sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, {left: -12, right: 12, top: 12, bottom: -12});
scene.add(sun);

// Court: surround, playing area, kitchen, lines, net.
function plane(w, l, color, cx, cy, y = 0) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, l), new THREE.MeshStandardMaterial({color, roughness: .95}));
  m.rotation.x = -Math.PI / 2; m.position.copy(P(cx, cy, y)); m.receiveShadow = true; scene.add(m); return m;
}
plane(W + 8, L + 8, 0x2f6b45, W / 2, L / 2, -0.002);
plane(W, L, 0x2456a6, W / 2, L / 2);
plane(W, C.nvz_far - C.nvz_near, 0x3a6fbf, W / 2, C.net_y, 0.001);
const lineMat = new THREE.MeshBasicMaterial({color: 0xffffff});
function line(x1, y1, x2, y2) {
  const a = P(x1, y1, .004), b = P(x2, y2, .004), len = a.distanceTo(b);
  const m = new THREE.Mesh(new THREE.PlaneGeometry(len, .05), lineMat);
  m.rotation.x = -Math.PI / 2; m.rotation.z = -Math.atan2(b.z - a.z, b.x - a.x);
  m.position.copy(a.clone().add(b).multiplyScalar(.5)); scene.add(m);
}
[[0,0,W,0],[0,L,W,L],[0,0,0,L],[W,0,W,L],[0,C.nvz_near,W,C.nvz_near],[0,C.nvz_far,W,C.nvz_far],
 [W/2,0,W/2,C.nvz_near],[W/2,C.nvz_far,W/2,L]].forEach(s => line(...s));
const net = new THREE.Mesh(new THREE.PlaneGeometry(W + .6, .86),
  new THREE.MeshStandardMaterial({color: 0x111111, transparent: true, opacity: .55, side: THREE.DoubleSide}));
net.position.copy(P(W / 2, C.net_y, .43)); scene.add(net);
const tape = new THREE.Mesh(new THREE.BoxGeometry(W + .6, .05, .03), new THREE.MeshStandardMaterial({color: 0xffffff}));
tape.position.copy(P(W / 2, C.net_y, .88)); scene.add(tape);
for (const x of [-.3, W + .3]) {
  const post = new THREE.Mesh(new THREE.CylinderGeometry(.03, .03, .92), new THREE.MeshStandardMaterial({color: 0x222222}));
  post.position.copy(P(x, C.net_y, .46)); post.castShadow = true; scene.add(post);
}

// Players.
const BONES = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const players = {};
const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 8);
const unitSph = new THREE.SphereGeometry(1, 12, 10);
function makePlayer(id) {
  const color = new THREE.Color(R.colors[id] || '#ffffff');
  const mat = new THREE.MeshStandardMaterial({color, roughness: .6});
  const g = new THREE.Group(); scene.add(g);
  const bones = BONES.map(() => { const m = new THREE.Mesh(unitCyl, mat); m.castShadow = true; g.add(m); return m; });
  const joints = Array.from({length: 17}, () => { const m = new THREE.Mesh(unitSph, mat); m.castShadow = true; g.add(m); return m; });
  // Fallback figure when no pose was found for this frame.
  const body = new THREE.Mesh(new THREE.CapsuleGeometry(.2, 1.1, 4, 10), mat); body.castShadow = true; g.add(body);
  const ring = new THREE.Mesh(new THREE.RingGeometry(.38, .46, 40), new THREE.MeshBasicMaterial({color, side: THREE.DoubleSide}));
  ring.rotation.x = -Math.PI / 2; g.add(ring);
  const trail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color, transparent: true, opacity: .8}));
  scene.add(trail);
  return players[id] = {g, bones, joints, body, ring, trail};
}
function setBone(m, a, b, r) {
  const d = b.clone().sub(a), len = d.length();
  m.position.copy(a).add(b).multiplyScalar(.5);
  m.scale.set(r, len, r);
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d.normalize());
}
const history = {};
R.frames.forEach((fr, i) => fr.players.forEach(p => (history[p.id] ||= []).push([i, p.x, p.y])));
function show(i) {
  const fr = R.frames[i]; if (!fr) return;
  const seen = new Set();
  for (const p of fr.players) {
    const pl = players[p.id] || makePlayer(p.id); seen.add(p.id);
    pl.g.visible = true; pl.trail.visible = true;
    pl.ring.position.copy(P(p.x, p.y, .01));
    const J = p.joints ? p.joints.map(j => j && P(j[0], j[1], j[2])) : null;
    const ok = J && J.filter(Boolean).length >= 8;
    pl.body.visible = !ok;
    if (!ok) pl.body.position.copy(P(p.x, p.y, .75));
    BONES.forEach(([a, b], k) => { const m = pl.bones[k]; m.visible = !!(ok && J[a] && J[b]); if (m.visible) setBone(m, J[a], J[b], .045); });
    pl.joints.forEach((m, k) => {
      m.visible = !!(ok && J[k] && (k === 0 || k >= 5));
      if (m.visible) { m.position.copy(J[k]); const r = k === 0 ? .13 : .06; m.scale.set(r, r, r); }
    });
    const h = history[p.id].filter(([j]) => j <= i && j > i - 3 * R.fps);
    pl.trail.geometry.setFromPoints(h.map(([, x, y]) => P(x, y, .02)));
  }
  for (const id in players) if (!seen.has(+id)) { players[id].g.visible = false; players[id].trail.visible = false; }
  document.getElementById('t').textContent = fr.t.toFixed(1) + ' s';
  seek.value = i;
}

// Camera presets.
const cam = R.camera;
const VIEWS = {
  'Broadcast': [P(W / 2, -9, 6.5), P(W / 2, L / 2, 0)],
  'Behind far baseline': [P(W / 2, L + 9, 6.5), P(W / 2, L / 2, 0)],
  'Side': [P(W + 11, L / 2, 5), P(W / 2, L / 2, 0)],
  'Top': [P(W / 2, L / 2 - .01, 19), P(W / 2, L / 2, 0)],
  'Match camera': [P(...cam.position), P(cam.position[0] + 10 * cam.forward[0], cam.position[1] + 10 * cam.forward[1], Math.max(0, cam.position[2] + 10 * cam.forward[2]))],
};
const views = document.getElementById('views');
let active = null;
function setView(name) {
  const [pos, tgt] = VIEWS[name];
  camera.position.copy(pos); controls.target.copy(tgt);
  camera.fov = name === 'Match camera' ? cam.fov_deg : 45; camera.updateProjectionMatrix(); controls.update();
  [...views.children].forEach(b => b.classList.toggle('on', b.textContent === name));
}
for (const name of [...Object.keys(VIEWS), 'Orbit']) {
  const b = document.createElement('button'); b.textContent = name;
  b.onclick = () => { if (name === 'Orbit') { controls.autoRotate = !controls.autoRotate; b.classList.toggle('on', controls.autoRotate); } else setView(name); };
  views.appendChild(b);
}
const legend = document.getElementById('legend');
const teamOf = {}; R.frames.forEach(f => f.players.forEach(p => teamOf[p.id] = p.team));
Object.keys(R.colors).sort((a, b) => a - b).forEach(id => {
  legend.insertAdjacentHTML('beforeend', `<div><span class="dot" style="background:${R.colors[id]}"></span>Player ${id} · ${teamOf[id]}</div>`);
});

// Playback.
const seek = document.getElementById('seek'); seek.max = Math.max(0, R.frames.length - 1);
let playing = true, speed = 1, cur = 0, last = performance.now();
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? '⏸' : '▶'; };
document.getElementById('speed').onclick = e => { speed = speed === 1 ? .5 : speed === .5 ? .25 : 1; e.target.textContent = speed + '×'; };
seek.oninput = () => { cur = +seek.value; show(cur); };
function resize() { const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
addEventListener('resize', resize); resize();
setView('Broadcast'); show(0);
renderer.setAnimationLoop(now => {
  if (playing && R.frames.length) {
    cur += (now - last) / 1000 * R.fps * speed;
    if (cur >= R.frames.length) cur = 0;
    show(Math.floor(cur));
  }
  last = now; controls.update(); renderer.render(scene, camera);
});
</script></body></html>"""


def replay_html(replay: dict) -> str:
    return TEMPLATE.replace("__REPLAY__", json.dumps(replay, separators=(",", ":")).replace("</", "<\\/"))
