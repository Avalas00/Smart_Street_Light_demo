const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const MAC_A = '0024E979D6';
const MAC_B = '0024E9D1FF';
const html = fs.readFileSync(path.join(__dirname, 'templates/index.html'), 'utf8');
const scripts = [...html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/gi)]
    .map(match => match[1]).filter(script => script.trim());

class Element {
    constructor() {
        this.style = {};
        this.textContent = '--';
        this.innerHTML = '';
        this.children = [];
        this.listeners = {};
        this.fields = {};
        this._value = '';
        this.classList = { add() {}, remove() {}, toggle() {} };
    }
    get value() { return this._value; }
    set value(value) { this._value = String(value); }
    addEventListener(name, listener) { this.listeners[name] = listener; }
    querySelector(selector) { return this.fields[selector] || null; }
    appendChild(child) { this.children.push(child); }
    removeChild(child) { this.children.splice(this.children.indexOf(child), 1); }
    insertAdjacentHTML(position, content) { this.innerHTML += content; }
}

function setup() {
    const elements = new Map([...html.matchAll(/\bid="([^"\s]+)"/g)]
        .map(match => [match[1], new Element()]));
    const cards = new Map();
    for (const mac of [MAC_A, MAC_B]) {
        const card = new Element();
        card.fields['.dev-brightness'] = new Element();
        cards.set(mac, card);
        const detail = new Element();
        detail.fields['[data-field="dimming"]'] = new Element();
        detail.fields['[data-field="voltage"]'] = new Element();
        elements.set(`dc-${mac}`, detail);
    }
    elements.get('device-select').value = MAC_A;
    const handlers = {};
    const sent = [];
    const timers = new Map();
    let nextTimer = 0;
    const socket = { connected: true, on: (name, handler) => { handlers[name] = handler; },
        emit: (name, data) => { sent.push({ name, data }); } };
    const context = vm.createContext({
        document: {
            getElementById: id => elements.get(id) || null,
            addEventListener() {},
            querySelectorAll: selector => {
                const match = selector.match(/^\[data-device-id="([^"]+)"\]$/);
                return match && cards.has(match[1]) ? [cards.get(match[1])] : [];
            },
            createElement: () => new Element(),
        },
        io: () => socket,
        fetch: () => new Promise(() => {}),
        window: { addEventListener() {} },
        AbortController,
        clearTimeout: id => timers.delete(id),
        setInterval() {},
        setTimeout: (callback, ms) => {
            const id = ++nextTimer;
            timers.set(id, { callback, ms });
            return id;
        },
    });
    for (const script of scripts) vm.runInContext(script, context);
    const run = code => vm.runInContext(code, context);
    const state = overrides => ({
        devices: [{ id: MAC_A, name: '79D6', group_id: 6 }, { id: MAC_B, name: 'D1FF', group_id: 6 }],
        device_brightness: { [MAC_A]: 23, [MAC_B]: 88 },
        device_brightness_ts: { [MAC_A]: 100, [MAC_B]: 100 },
        device_detail: {}, device_lost: {}, brightness: 10, mode: 'manual', ...overrides,
    });
    return { elements, cards, handlers, run, state, sent, socket, timers,
        render: data => context.renderState(data),
        value: mac => cards.get(mac).fields['.dev-brightness'].textContent,
        dimming: mac => elements.get(`dc-${mac}`).fields['[data-field="dimming"]'].textContent };
}

test('first state initializes per-node cache and displays the selected reported value', () => {
    const ui = setup();
    ui.render(ui.state());
    assert.equal(ui.run(`els.brightnessData['${MAC_A}']`), 23);
    assert.equal(ui.elements.get('brightness-slider').value, '23');
    assert.match(ui.elements.get('device-stats').innerHTML, />23%<\/strong>/);
    assert.match(ui.elements.get('device-stats').innerHTML, />88%<\/strong>/);
});

test('socket report patches only the matching MAC in overview and detail views', () => {
    const ui = setup();
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 62, ts: 100 });
    assert.equal(ui.value(MAC_A), '62%');
    assert.equal(ui.dimming(MAC_A), '62%');
    assert.equal(ui.value(MAC_B), '--');
    assert.equal(ui.elements.get('brightness-slider').value, '62');
});

test('another node report does not replace the selected node slider', () => {
    const ui = setup();
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 20, ts: 100 });
    ui.handlers.device_brightness_update({ device_id: MAC_B, brightness: 80, ts: 100 });
    assert.equal(ui.elements.get('brightness-slider').value, '20');
    ui.elements.get('device-select').value = MAC_B;
    ui.elements.get('device-select').listeners.change();
    assert.equal(ui.elements.get('brightness-slider').value, '80');
});

test('incoming report updates cards but preserves an unsent manual slider value', () => {
    const ui = setup();
    ui.elements.get('brightness-slider').value = 66;
    ui.elements.get('brightness-slider').listeners.input();
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 20, ts: 100 });
    assert.equal(ui.value(MAC_A), '20%');
    assert.equal(ui.elements.get('brightness-slider').value, '66');
    ui.render(ui.state());
    assert.equal(ui.elements.get('brightness-slider').value, '66');
});

test('older snapshots and detail events cannot overwrite a more recent report', () => {
    const ui = setup();
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 65, ts: 200 });
    ui.render(ui.state());
    ui.handlers.device_detail_update({ device_id: MAC_A, dimming: 10, ts: 100 });
    assert.equal(ui.run(`els.brightnessData['${MAC_A}']`), 65);
    assert.equal(ui.value(MAC_A), '65%');
    assert.equal(ui.dimming(MAC_A), '65%');
    assert.match(ui.elements.get('device-stats').innerHTML, />65%<\/strong>/);
});

test('unknown brightness is not shown as zero, but a real zero report is shown', () => {
    const ui = setup();
    ui.render(ui.state({ device_brightness: { [MAC_A]: null, [MAC_B]: null } }));
    assert.doesNotMatch(ui.elements.get('device-stats').innerHTML, /0%/);
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 0, ts: 101 });
    assert.equal(ui.value(MAC_A), '0%');
    assert.equal(ui.elements.get('brightness-slider').value, '0');
});

test('state snapshots restore detail cards as well as overview brightness', () => {
    const ui = setup();
    ui.render(ui.state({ device_detail: {
        [MAC_A]: { device_id: MAC_A, dimming: 23, voltage: 115.4, ts: 100 },
    } }));
    assert.equal(ui.dimming(MAC_A), '23%');
    assert.equal(ui.elements.get(`dc-${MAC_A}`).fields['[data-field="voltage"]'].textContent, '115.4');
});

test('invalid brightness does not replace a real report', () => {
    const ui = setup();
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 45, ts: 100 });
    for (const brightness of [null, '10', -1, 101]) {
        ui.handlers.device_brightness_update({ device_id: MAC_A, brightness, ts: 101 });
    }
    assert.equal(ui.value(MAC_A), '45%');
});

test('group control keeps its target instead of adopting one node report', () => {
    const ui = setup();
    ui.run("_ctrlTarget = 'group'");
    ui.elements.get('brightness-slider').value = 60;
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 20, ts: 100 });
    assert.equal(ui.value(MAC_A), '20%');
    assert.equal(ui.elements.get('brightness-slider').value, '60');
});

function sendSingle(ui, brightness) {
    ui.elements.get('brightness-slider').value = brightness;
    ui.elements.get('brightness-slider').listeners.input();
    ui.run('sendBrightness()');
}

test('sending a single-node value is not reset by the previous zero state', () => {
    const ui = setup();
    const oldState = ui.state({ brightness: 0, device_brightness: { [MAC_A]: 0, [MAC_B]: 0 } });
    ui.render(oldState);
    sendSingle(ui, 70);
    assert.equal(ui.sent[0].data.brightness, 70);
    assert.equal(ui.sent[0].data.device_id, MAC_A);
    ui.render(oldState);
    assert.equal(ui.elements.get('brightness-slider').value, '70');
    assert.equal(ui.value(MAC_A), '0%');
});

test('a different reported value cannot clear a command waiting for confirmation', () => {
    const ui = setup();
    ui.render(ui.state());
    sendSingle(ui, 70);
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 0, ts: 101 });
    assert.equal(ui.elements.get('brightness-slider').value, '70');
    assert.equal(ui.value(MAC_A), '0%');
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 70, ts: 102 });
    assert.equal(ui.run('_pendingBrightness'), null);
    assert.equal(ui.value(MAC_A), '70%');
});

test('only a newer report for the same MAC confirms the submitted value', () => {
    const ui = setup();
    ui.render(ui.state());
    sendSingle(ui, 23);
    ui.render(ui.state());
    assert.equal(ui.run('_pendingBrightness'), 23);
    ui.handlers.device_brightness_update({ device_id: MAC_B, brightness: 23, ts: 101 });
    assert.equal(ui.run('_pendingBrightness'), 23);
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 23, ts: 101 });
    assert.equal(ui.run('_pendingBrightness'), null);
});

test('confirmation of a previous command preserves a newer unsent edit', () => {
    const ui = setup();
    ui.render(ui.state());
    sendSingle(ui, 70);
    ui.elements.get('brightness-slider').value = 90;
    ui.elements.get('brightness-slider').listeners.input();
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 70, ts: 101 });
    assert.equal(ui.elements.get('brightness-slider').value, '90');
    assert.equal(ui.run('_pendingBrightness'), 90);
});

test('switching nodes discards the old control wait but keeps node reports independent', () => {
    const ui = setup();
    ui.render(ui.state());
    sendSingle(ui, 70);
    ui.elements.get('device-select').value = MAC_B;
    ui.elements.get('device-select').listeners.change();
    assert.equal(ui.elements.get('brightness-slider').value, '88');
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 70, ts: 101 });
    assert.equal(ui.elements.get('brightness-slider').value, '88');
});

test('an explicit zero command remains valid while waiting for its report', () => {
    const ui = setup();
    ui.render(ui.state());
    sendSingle(ui, 0);
    ui.render(ui.state());
    assert.equal(ui.elements.get('brightness-slider').value, '0');
    assert.equal(ui.sent[0].data.brightness, 0);
    ui.handlers.device_brightness_update({ device_id: MAC_A, brightness: 0, ts: 101 });
    assert.equal(ui.run('_pendingBrightness'), null);
});

test('socket reconnect restores controls and reopens the video without reloading the page', () => {
    const ui = setup();
    ui.render(ui.state());
    ui.socket.connected = false;
    ui.handlers.disconnect('transport close');
    ui.run('sendBrightness()');
    assert.equal(ui.sent.length, 0);
    assert.equal(ui.elements.get('live-status').hidden, false);
    assert.equal(ui.elements.get('send-btn').disabled, true);
    ui.socket.connected = true;
    ui.handlers.connect();
    assert.equal(ui.elements.get('live-status').hidden, true);
    assert.equal(ui.elements.get('send-btn').disabled, false);
    assert.match(ui.elements.get('live-video').src, /^\/video_feed\?reconnect=/);
});

test('video errors schedule exactly one retry even while the live socket stays connected', () => {
    const ui = setup();
    const video = ui.elements.get('live-video');
    video.listeners.error();
    video.listeners.error();
    const retries = [...ui.timers.values()].filter(timer => timer.ms === 2000);
    assert.equal(retries.length, 1);
    retries[0].callback();
    assert.match(video.src, /^\/video_feed\?reconnect=/);
});

test('the emergency banner disables operator controls and restores them after release', () => {
    const ui = setup();
    ui.render(ui.state({emergency_trigger:true, level:4, brightness:100}));
    assert.equal(ui.elements.get('emergency-banner').style.display, 'flex');
    assert.equal(ui.elements.get('send-btn').disabled, true);
    assert.equal(ui.elements.get('manual-btn').disabled, true);
    ui.run('sendBrightness(); setMode("auto"); setGroup()');
    assert.equal(ui.sent.length, 0);
    ui.render(ui.state({emergency_trigger:false, level:2, brightness:30}));
    assert.equal(ui.elements.get('emergency-banner').style.display, 'none');
    assert.equal(ui.elements.get('send-btn').disabled, false);
});
