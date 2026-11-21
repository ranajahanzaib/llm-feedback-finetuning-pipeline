import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CHROME_PATH =
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9222;
const SCREENSHOT_DIR = path.resolve("docs/screenshots");

fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function getDebuggerUrl() {
  for (let i = 0; i < 30; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      if (res.ok) {
        const list = await res.json();
        const page = list.find((item) => item.type === "page");
        if (page && page.webSocketDebuggerUrl) {
          return page.webSocketDebuggerUrl;
        }
      }
    } catch {
      // Chrome starting up
    }
    await sleep(200);
  }
  throw new Error("Could not connect to Chrome debugging port");
}

class CDPClient {
  constructor(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.id = 1;
    this.callbacks = new Map();
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.ws.onopen = resolve;
      this.ws.onerror = reject;
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id && this.callbacks.has(msg.id)) {
          const cb = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) cb.reject(new Error(msg.error.message));
          else cb.resolve(msg.result);
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = this.id++;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    this.ws.close();
  }
}

async function run() {
  console.log("Launching headless Chrome...");
  const chromeProcess = spawn(
    CHROME_PATH,
    [
      "--headless=new",
      `--remote-debugging-port=${PORT}`,
      "--disable-gpu",
      "--no-sandbox",
      "--window-size=1440,900",
      "http://127.0.0.1:5173/",
    ],
    { stdio: "ignore" },
  );

  try {
    const wsUrl = await getDebuggerUrl();
    console.log("Connected to Chrome:", wsUrl);
    const client = new CDPClient(wsUrl);
    await client.connect();

    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Emulation.setDeviceMetricsOverride", {
      width: 1440,
      height: 900,
      deviceScaleFactor: 2,
      mobile: false,
    });

    console.log("Waiting for app to load...");
    await sleep(2500);

    async function screenshot(name) {
      const res = await client.send("Page.captureScreenshot", {
        format: "png",
        captureBeyondViewport: false,
      });
      const filePath = path.join(SCREENSHOT_DIR, `${name}.png`);
      fs.writeFileSync(filePath, Buffer.from(res.data, "base64"));
      console.log(`Saved screenshot: ${filePath}`);
    }

    async function clickNav(label) {
      await client.send("Runtime.evaluate", {
        expression: `(() => {
          const btns = Array.from(document.querySelectorAll('nav button'));
          const btn = btns.find(b => b.textContent.includes('${label}'));
          if (btn) btn.click();
        })()`,
      });
      await sleep(1000);
    }

    // 1. Overview
    console.log("Capturing overview...");
    await screenshot("overview");

    // 2. Corrections
    console.log("Capturing corrections...");
    await clickNav("Corrections");
    await screenshot("corrections");

    // 3. Datasets
    console.log("Capturing datasets...");
    await clickNav("Datasets");
    await screenshot("datasets");

    // 4. Evaluation
    console.log("Capturing evaluation...");
    await clickNav("Evaluation");
    await screenshot("evaluation");

    // 5. Registry
    console.log("Capturing registry...");
    await clickNav("Registry");
    await screenshot("registry");

    // 6. Playground
    console.log("Capturing playground...");
    await clickNav("Playground");
    // Trigger prediction in playground
    await client.send("Runtime.evaluate", {
      expression: `(() => {
        const btn = document.querySelector('.playground button.primary');
        if (btn) btn.click();
      })()`,
    });
    await sleep(1500);
    await screenshot("playground");

    // 7. Experiments
    console.log("Capturing experiments...");
    await clickNav("Experiments");
    await screenshot("experiments");

    client.close();
    console.log("All screenshots captured successfully!");
  } finally {
    chromeProcess.kill();
  }
}

run().catch((err) => {
  console.error("Error capturing screenshots:", err);
  process.exit(1);
});
