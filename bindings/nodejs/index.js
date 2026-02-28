/**
 * PyAccelerate Node.js Client
 *
 * Communicates with the PyAccelerate HTTP API server.
 * Start the server first: `pyaccelerate serve --http-port 8420`
 *
 * @example
 * const { PyAccelerate } = require('pyaccelerate');
 * const client = new PyAccelerate('http://localhost:8420');
 *
 * const info = await client.getInfo();
 * const cpuData = await client.getCPU();
 * const gpuData = await client.getGPU();
 * const metrics = await client.getMetrics();
 */

"use strict";

const http = require("http");
const https = require("https");

class PyAccelerateError extends Error {
  constructor(message, statusCode, body) {
    super(message);
    this.name = "PyAccelerateError";
    this.statusCode = statusCode;
    this.body = body;
  }
}

class PyAccelerate {
  /**
   * Create a new PyAccelerate client.
   * @param {string} [baseUrl='http://localhost:8420'] - Server base URL
   * @param {object} [options] - Additional options
   * @param {number} [options.timeout=30000] - Request timeout in ms
   * @param {object} [options.headers] - Extra headers to send
   */
  constructor(baseUrl = "http://localhost:8420", options = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeout = options.timeout || 30000;
    this.headers = options.headers || {};
  }

  /**
   * Make an HTTP request to the server.
   * @private
   */
  _request(method, path, body = null) {
    return new Promise((resolve, reject) => {
      const url = new URL(`${this.baseUrl}${path}`);
      const isHttps = url.protocol === "https:";
      const mod = isHttps ? https : http;

      const headers = {
        Accept: "application/json",
        ...this.headers,
      };

      let bodyStr = null;
      if (body !== null) {
        bodyStr = JSON.stringify(body);
        headers["Content-Type"] = "application/json";
        headers["Content-Length"] = Buffer.byteLength(bodyStr);
      }

      const req = mod.request(
        {
          hostname: url.hostname,
          port: url.port,
          path: url.pathname + url.search,
          method,
          headers,
          timeout: this.timeout,
        },
        (res) => {
          let data = "";
          res.on("data", (chunk) => (data += chunk));
          res.on("end", () => {
            if (res.statusCode >= 400) {
              let parsed;
              try {
                parsed = JSON.parse(data);
              } catch {
                parsed = data;
              }
              reject(
                new PyAccelerateError(
                  `HTTP ${res.statusCode}: ${parsed.error || data}`,
                  res.statusCode,
                  parsed
                )
              );
              return;
            }

            // /metrics returns plain text
            const ct = res.headers["content-type"] || "";
            if (ct.includes("application/json")) {
              try {
                resolve(JSON.parse(data));
              } catch (e) {
                reject(new PyAccelerateError(`Invalid JSON: ${e.message}`));
              }
            } else {
              resolve(data);
            }
          });
        }
      );

      req.on("error", (err) =>
        reject(new PyAccelerateError(`Connection failed: ${err.message}`))
      );
      req.on("timeout", () => {
        req.destroy();
        reject(new PyAccelerateError("Request timed out"));
      });

      if (bodyStr) req.write(bodyStr);
      req.end();
    });
  }

  // ── API Methods ──────────────────────────────────────────────────────

  /** Full hardware info (machine-readable dict). */
  async getInfo() {
    return this._request("GET", "/api/v1/info");
  }

  /** Human-readable hardware summary. */
  async getSummary() {
    return this._request("GET", "/api/v1/summary");
  }

  /** CPU details. */
  async getCPU() {
    return this._request("GET", "/api/v1/cpu");
  }

  /** GPU details. */
  async getGPU() {
    return this._request("GET", "/api/v1/gpu");
  }

  /** NPU details. */
  async getNPU() {
    return this._request("GET", "/api/v1/npu");
  }

  /** Memory stats. */
  async getMemory() {
    return this._request("GET", "/api/v1/memory");
  }

  /** Virtualization info. */
  async getVirt() {
    return this._request("GET", "/api/v1/virt");
  }

  /** One-line status. */
  async getStatus() {
    return this._request("GET", "/api/v1/status");
  }

  /** Prometheus metrics (plain text). */
  async getMetrics() {
    return this._request("GET", "/api/v1/metrics");
  }

  /**
   * Run benchmarks.
   * @param {boolean} [full=false] - Run full (slower) benchmark suite
   */
  async runBenchmark(full = false) {
    if (full) {
      return this._request("POST", "/api/v1/benchmark", { full: true });
    }
    return this._request("GET", "/api/v1/benchmark");
  }

  /** Run auto-tune and return the profile. */
  async runTune() {
    return this._request("POST", "/api/v1/tune");
  }

  /** Get the current tune profile. */
  async getTuneProfile() {
    return this._request("GET", "/api/v1/tune/profile");
  }

  /** Health check. Returns 'OK' if server is running. */
  async health() {
    return this._request("GET", "/health");
  }

  /** List available API endpoints. */
  async listEndpoints() {
    return this._request("GET", "/api/v1");
  }
}

module.exports = { PyAccelerate, PyAccelerateError };
