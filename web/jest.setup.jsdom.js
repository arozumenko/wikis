// Polyfill TextEncoder / TextDecoder for jest-environment-jsdom
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;
