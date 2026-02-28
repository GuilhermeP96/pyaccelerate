/**
 * Basic test for PyAccelerate Node.js client.
 *
 * Requires the server to be running:
 *   pyaccelerate serve --http-port 8420
 */

const { PyAccelerate, PyAccelerateError } = require("./index");

const BASE = process.env.PYACCELERATE_URL || "http://localhost:8420";
const client = new PyAccelerate(BASE, { timeout: 60000 });

async function test(name, fn) {
  try {
    await fn();
    console.log(`  ✓ ${name}`);
  } catch (err) {
    console.error(`  ✗ ${name}: ${err.message}`);
    process.exitCode = 1;
  }
}

(async () => {
  console.log(`\nPyAccelerate Node.js client tests (${BASE})\n`);

  await test("health", async () => {
    const res = await client.health();
    if (res !== "OK") throw new Error(`Expected "OK", got "${res}"`);
  });

  await test("listEndpoints", async () => {
    const res = await client.listEndpoints();
    if (!res.endpoints) throw new Error("Missing endpoints");
    if (!res.endpoints.GET.length) throw new Error("No GET endpoints");
  });

  await test("getInfo", async () => {
    const info = await client.getInfo();
    if (!info.cpu) throw new Error("Missing cpu");
    if (!info.gpu) throw new Error("Missing gpu");
  });

  await test("getSummary", async () => {
    const res = await client.getSummary();
    if (!res.summary) throw new Error("Missing summary");
  });

  await test("getCPU", async () => {
    const cpu = await client.getCPU();
    if (!cpu.brand) throw new Error("Missing brand");
    if (cpu.physical_cores < 1) throw new Error("Invalid core count");
  });

  await test("getGPU", async () => {
    const gpu = await client.getGPU();
    if (gpu.count === undefined) throw new Error("Missing count");
  });

  await test("getNPU", async () => {
    const npu = await client.getNPU();
    if (npu.count === undefined) throw new Error("Missing count");
  });

  await test("getMemory", async () => {
    const mem = await client.getMemory();
    if (!mem.pressure) throw new Error("Missing pressure");
  });

  await test("getVirt", async () => {
    const virt = await client.getVirt();
    if (virt.inside_container === undefined)
      throw new Error("Missing inside_container");
  });

  await test("getStatus", async () => {
    const res = await client.getStatus();
    if (!res.status) throw new Error("Missing status");
  });

  await test("getMetrics", async () => {
    const metrics = await client.getMetrics();
    if (!metrics.includes("pyaccelerate_")) throw new Error("Invalid metrics");
  });

  console.log("\nDone.\n");
})();
