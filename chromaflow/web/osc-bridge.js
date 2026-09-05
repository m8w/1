#!/usr/bin/env node
'use strict';
/*
 * Chromaflow OSC bridge.
 *
 * Browsers cannot open raw UDP sockets, so Chromaflow's "Sync out (OSC)"
 * feature (Input tab) sends its OSC messages over a WebSocket instead. This
 * script is the other half: it accepts that WebSocket connection and forwards
 * every message, byte for byte, out as real OSC-over-UDP — to an
 * openFrameworks sketch (ofxOsc), TouchDesigner, Max/MSP, Pure Data, or
 * anything else that speaks OSC.
 *
 * No npm install, no package.json, no dependencies — only Node's own http,
 * crypto and dgram modules. Requires nothing but Node itself.
 *
 * Usage:
 *   node osc-bridge.js [--ws-port 8081] [--osc-host 127.0.0.1] [--osc-port 9000]
 *
 * Then in Chromaflow, Input tab -> Sync out (OSC) -> Connect, using
 * ws://127.0.0.1:8081 (or whatever --ws-port you chose).
 */
const http = require('http');
const crypto = require('crypto');
const dgram = require('dgram');

function arg(name, fallback) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}
const WS_PORT = parseInt(arg('ws-port', '8081'), 10);
const OSC_HOST = arg('osc-host', '127.0.0.1');
const OSC_PORT = parseInt(arg('osc-port', '9000'), 10);

const WS_MAGIC = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11';
const udp = dgram.createSocket('udp4');

/* OSC messages start with a null-terminated ASCII address; pulling it out is
   just for the console log, so a receiver's own decoder is what actually matters. */
function oscAddress(buf) {
  const end = buf.indexOf(0);
  return end > 0 ? buf.toString('ascii', 0, end) : '(unrecognised)';
}

function sendFrame(socket, opcode, payload) {
  payload = payload || Buffer.alloc(0);
  const len = payload.length;
  let header;
  if (len < 126) {
    header = Buffer.from([0x80 | opcode, len]);
  } else if (len < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x80 | opcode; header[1] = 126;
    header.writeUInt16BE(len, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x80 | opcode; header[1] = 127;
    header.writeBigUInt64BE(BigInt(len), 2);
  }
  socket.write(Buffer.concat([header, payload]));
}

function handleFrame(opcode, payload, socket) {
  if (opcode === 0x8) { socket.end(); return; }                    // close
  if (opcode === 0x9) { sendFrame(socket, 0xA, payload); return; } // ping -> pong
  if (opcode !== 0x2 && opcode !== 0x1) return;                    // only binary/text carry OSC
  udp.send(payload, OSC_PORT, OSC_HOST, err => {
    if (err) console.error('UDP send failed:', err.message);
  });
  console.log('→ udp://' + OSC_HOST + ':' + OSC_PORT + '  ' + oscAddress(payload));
}

/* A minimal RFC 6455 frame parser. Client-to-server frames are always masked;
   fragmentation is not reassembled since every message Chromaflow sends fits
   in one frame — a fragmented frame's later parts are simply dropped, which
   never happens in practice for this bridge's traffic. */
function makeParser(socket) {
  let buf = Buffer.alloc(0);
  return chunk => {
    buf = Buffer.concat([buf, chunk]);
    for (;;) {
      if (buf.length < 2) return;
      const b0 = buf[0], b1 = buf[1];
      const fin = (b0 & 0x80) !== 0;
      const opcode = b0 & 0x0f;
      const masked = (b1 & 0x80) !== 0;
      let len = b1 & 0x7f;
      let off = 2;
      if (len === 126) {
        if (buf.length < 4) return;
        len = buf.readUInt16BE(2); off = 4;
      } else if (len === 127) {
        if (buf.length < 10) return;
        len = Number(buf.readBigUInt64BE(2)); off = 10;
      }
      let maskKey = null;
      if (masked) {
        if (buf.length < off + 4) return;
        maskKey = buf.slice(off, off + 4); off += 4;
      }
      if (buf.length < off + len) return; // wait for the rest of this frame
      let payload = buf.slice(off, off + len);
      if (maskKey) {
        payload = Buffer.from(payload); // copy first — unmasking in place would corrupt buf
        for (let i = 0; i < payload.length; i++) payload[i] ^= maskKey[i % 4];
      }
      buf = buf.slice(off + len);
      if (fin) handleFrame(opcode, payload, socket);
    }
  };
}

const server = http.createServer((req, res) => { res.writeHead(404); res.end(); });
server.on('upgrade', (req, socket) => {
  const key = req.headers['sec-websocket-key'];
  if (!key) { socket.destroy(); return; }
  const accept = crypto.createHash('sha1').update(key + WS_MAGIC).digest('base64');
  socket.write(
    'HTTP/1.1 101 Switching Protocols\r\n' +
    'Upgrade: websocket\r\n' +
    'Connection: Upgrade\r\n' +
    'Sec-WebSocket-Accept: ' + accept + '\r\n\r\n'
  );
  console.log('Chromaflow connected from', req.socket.remoteAddress);
  socket.on('data', makeParser(socket));
  socket.on('error', () => {});
  socket.on('close', () => console.log('Chromaflow disconnected'));
});

server.listen(WS_PORT, () => {
  console.log('Chromaflow OSC bridge');
  console.log('  listening for Chromaflow on ws://127.0.0.1:' + WS_PORT);
  console.log('  forwarding OSC to        udp://' + OSC_HOST + ':' + OSC_PORT);
});
