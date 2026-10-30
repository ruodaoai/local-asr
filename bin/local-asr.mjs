#!/usr/bin/env node
/**
 * local-asr —— 若道「本地转写引擎」插件脚本（双引擎）
 *
 * 引擎选择（RUODAO_ASR_ENGINE，默认 whisper）：
 *   whisper  faster-whisper-server（OpenAI 兼容，多语言，small≈460MB）
 *   funasr   FunASR Paraformer-large onnx（中文最强开源，CPU 无 torch，模型≈230MB）
 *
 * 命令（两引擎共用同一套流程与接口）：
 *   install  一键安装：venv + 依赖 + 模型下载（幂等，可续装）
 *   start    后台启动本地转写服务（OpenAI 兼容 /v1/audio/transcriptions）
 *   stop     停止服务（两引擎都会尝试停止）
 *   status   查看安装/运行状态 + 安装进度 + 日志尾部
 *   test     生成测试音频并转写，验证接口
 *   apply-config  一键配置会议助手（探测设备 + 写 local 通道配置）
 *
 * 环境变量：
 *   RUODAO_ASR_PORT    服务端口，默认 8000
 *   RUODAO_ASR_ENGINE  whisper | funasr（默认 whisper）
 *   RUODAO_ASR_MODEL   whisper 引擎模型规格 small|base|medium
 *   RUODAO_ASR_HOME    安装目录，默认 D:\ruodao-local-asr（无 D 盘回退用户目录）
 *   RUODAO_ASR_PYTHON  指定 Python 解释器
 *
 * 安装完成后服务地址 http://127.0.0.1:<port>/v1，模型名固定 whisper-1
 * （FunASR 服务对 model 参数不敏感，同样接受 whisper-1，会议助手零改动）。
 *
 * 安装进度：install.log（完整日志）与 install-state.json（阶段/结果）写在安装目录，
 * status 命令会带上两者，工作台据此展示实时进度；进程被强制结束时按 pid 判断为中断。
 */

import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PLUGIN_ENV_KEYS = ["RUODAO_ASR_ENGINE", "RUODAO_ASR_HOME", "RUODAO_ASR_PORT", "RUODAO_ASR_MODEL", "RUODAO_ASR_PYTHON"];
// 宿主（桌面端）会把 payload 以 JSON 写入 stdin 后立即 end()。
// 异步读取 + 超时兜底：宿主未写 stdin 或调用方保持管道打开时不会卡死。
function readPluginConfig(timeoutMs = 3000) {
  if (process.stdin.isTTY) return Promise.resolve(null);
  return new Promise((resolve) => {
    let buf = "";
    let settled = false;
    const finish = (payload) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      try { process.stdin.removeListener("data", onData); process.stdin.removeListener("end", onEnd); } catch { /* 忽略 */ }
      try { process.stdin.pause(); } catch { /* 忽略 */ }
      resolve(payload);
    };
    const timer = setTimeout(() => finish(null), timeoutMs);
    const parse = () => {
      const raw = buf.trim();
      if (!raw) return finish(null);
      try {
        const payload = JSON.parse(raw);
        finish(payload && typeof payload.config === "object" ? payload.config : null);
      } catch { finish(null); }
    };
    const onData = (d) => { buf += String(d); };
    const onEnd = () => parse();
    process.stdin.on("data", onData);
    process.stdin.on("end", onEnd);
    try { process.stdin.resume(); } catch { /* 忽略 */ }
  });
}

function applyPluginConfig(cfg) {
  if (cfg && typeof cfg === "object") {
    for (const key of PLUGIN_ENV_KEYS) {
      if (typeof cfg[key] === "string" && cfg[key].trim()) process.env[key] = cfg[key].trim();
    }
  }
  // 安装目录默认优先 D 盘（D:\ruodao-local-asr），D 盘不存在时回退用户目录。
  HOME = process.env.RUODAO_ASR_HOME || (fs.existsSync("D:/")
    ? "D:\\ruodao-local-asr"
    : path.join(os.homedir(), ".ruodao-local-asr"));
  PORT = Number(process.env.RUODAO_ASR_PORT || 8000);
  ENGINE = (process.env.RUODAO_ASR_ENGINE || "whisper").toLowerCase() === "funasr" ? "funasr" : "whisper";
  BASE_URL = `http://127.0.0.1:${PORT}`;
  MODEL_SPEC = (process.env.RUODAO_ASR_MODEL || "small").toLowerCase();
  HF_MODEL = MODEL_MAP[MODEL_SPEC] || MODEL_MAP.small;
}

// —— whisper 引擎（faster-whisper-server）——
let HOME, PORT, ENGINE, MODEL_SPEC, HF_MODEL, BASE_URL;
const MODEL_MAP = {
  small: "Systran/faster-whisper-small",     // ~460MB，推荐
  base: "Systran/faster-whisper-base",       // ~140MB，速度快
  medium: "Systran/faster-whisper-medium",   // ~1.5GB，准确率高
};
let VENV, MODELS, PY, SERVER_CMD, CONFIG_YAML;
let WHISPER_LOG, WHISPER_PID_FILE, INSTALL_LOG, INSTALL_STATE;

function refreshPaths() {
  VENV = path.join(HOME, "venv");
  MODELS = path.join(HOME, "models");
  PY = path.join(VENV, "Scripts", "python.exe");
  // 注意：不要用 faster-whisper-server.exe 启动 —— pip 生成的 exe 启动器内嵌 venv 绝对路径，
  // venv 迁移/部分环境下会静默失效，必须用 venv 的 python 启动。
  SERVER_CMD = [PY, "-c", "from faster_whisper_server.cli import main; main()"];
  CONFIG_YAML = path.join(MODELS, "server-config.yaml");
  WHISPER_LOG = path.join(HOME, "whisper-server.log");
  WHISPER_PID_FILE = path.join(HOME, "whisper-server.pid");
  INSTALL_LOG = path.join(HOME, "install.log");
  INSTALL_STATE = path.join(HOME, "install-state.json");
  FUNASR_VENV = path.join(HOME, "funasr-venv");
  FUNASR_PY = path.join(FUNASR_VENV, "Scripts", "python.exe");
  FUNASR_MODELS = path.join(HOME, "funasr-models");
  FUNASR_MODEL_DIR = path.join(FUNASR_MODELS, "paraformer-large-onnx");
  FUNASR_SERVER_DST = path.join(HOME, "funasr_server.py");
  FUNASR_FLAG = path.join(HOME, "funasr-installed.flag");
  FUNASR_PID_FILE = path.join(HOME, "funasr-server.pid");
  FUNASR_LOG = path.join(HOME, "funasr-server.log");
}

// —— funasr 引擎（funasr-onnx Paraformer）——
let FUNASR_VENV, FUNASR_PY, FUNASR_MODELS, FUNASR_MODEL_DIR, FUNASR_SERVER_DST, FUNASR_FLAG, FUNASR_PID_FILE, FUNASR_LOG;
const FUNASR_MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
// 仓库实际发布的是量化模型 model_quant.onnx（Paraformer 以 quantize=True 加载）。
const FUNASR_MODEL_FILES = ["model_quant.onnx", "model.onnx"];
const FUNASR_SERVER_SRC = path.join(path.dirname(fileURLToPath(import.meta.url)), "funasr_server.py");

const out = (obj) => process.stdout.write(JSON.stringify(obj, null, 2) + "\n");
const engineLabel = (engine = ENGINE) => (engine === "funasr" ? "FunASR Paraformer" : "faster-whisper");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/* ---------------- 安装进度（install.log / install-state.json） ---------------- */

function appendInstallLog(text) {
  if (!text) return;
  try {
    fs.mkdirSync(HOME, { recursive: true });
    fs.appendFileSync(INSTALL_LOG, text, "utf8");
  } catch { /* 日志失败不影响安装 */ }
}

function resetInstallLog() {
  try {
    fs.mkdirSync(HOME, { recursive: true });
    fs.writeFileSync(INSTALL_LOG, "", "utf8");
  } catch { /* 忽略 */ }
}

function readInstallState() {
  try { return JSON.parse(fs.readFileSync(INSTALL_STATE, "utf8")); } catch { return null; }
}

function writeInstallState(patch) {
  try {
    const next = { ...(readInstallState() || {}), ...patch, engine: patch.engine || ENGINE, updatedAt: new Date().toISOString() };
    fs.mkdirSync(HOME, { recursive: true });
    // 原子写：status 可能在安装过程中并发读取
    const tmp = `${INSTALL_STATE}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(next, null, 2), "utf8");
    fs.renameSync(tmp, INSTALL_STATE);
  } catch { /* 状态写入失败不影响安装 */ }
}

function setInstallPhase(phase, message) {
  writeInstallState({ running: true, phase, message });
  appendInstallLog(`\n[local-asr] ${message}\n`);
}

function pidAlive(pid) {
  const value = Number(pid);
  if (!Number.isInteger(value) || value <= 0) return false;
  try { process.kill(value, 0); return true; } catch (error) { return error?.code === "EPERM"; }
}

function readLogTail(file, maxBytes = 16000) {
  try {
    const { size } = fs.statSync(file);
    if (!size) return "";
    const start = Math.max(0, size - maxBytes);
    const fd = fs.openSync(file, "r");
    const buf = Buffer.alloc(size - start);
    try { fs.readSync(fd, buf, 0, buf.length, start); } finally { fs.closeSync(fd); }
    let text = buf.toString("utf8");
    if (start > 0) {
      const firstBreak = text.indexOf("\n");
      if (firstBreak >= 0) text = text.slice(firstBreak + 1);
    }
    return text.replace(/\r(?!\n)/g, "\n");
  } catch { return ""; }
}

function installSnapshot() {
  const state = readInstallState();
  const snapshot = { ...(state || {}) };
  const alive = Boolean(state?.running && pidAlive(state.pid));
  if (state?.running && !alive) {
    snapshot.running = false;
    snapshot.stale = true;
    snapshot.message = state.message ? `${state.message}（安装进程已退出）` : "上次安装未完成（安装进程已退出）";
  } else {
    snapshot.running = alive;
  }
  if (snapshot.startedAt && !snapshot.finishedAt && Number.isFinite(Date.parse(snapshot.startedAt))) {
    snapshot.elapsedSeconds = Math.max(0, Math.round((Date.now() - Date.parse(snapshot.startedAt)) / 1000));
  }
  snapshot.log = readLogTail(INSTALL_LOG);
  snapshot.logFile = INSTALL_LOG;
  return snapshot;
}

/* ---------------- 环境探测 ---------------- */

function versionOf(python) {
  const r = spawnSync(python, ["--version"], { encoding: "utf8", windowsHide: true });
  return String(`${r.stdout || ""}${r.stderr || ""}`).trim();
}

function findPython() {
  const candidates = [process.env.RUODAO_ASR_PYTHON, "python", "py"].filter(Boolean);
  for (const cmd of candidates) {
    const r = spawnSync(cmd, ["--version"], { encoding: "utf8", windowsHide: true });
    if (r.status === 0) return cmd;
  }
  return null;
}

// Python 探测成本约 100-300ms；成功结果缓存 10 分钟，失败不缓存（装好 Python 后立即生效）。
function probePython() {
  const cacheFile = path.join(HOME, "python-probe.json");
  try {
    const cached = JSON.parse(fs.readFileSync(cacheFile, "utf8"));
    if (cached?.available && Date.now() - Number(cached.at || 0) < 10 * 60 * 1000) return cached;
  } catch { /* 无缓存 */ }
  const cmd = findPython();
  const result = cmd
    ? { available: true, command: cmd, version: versionOf(cmd), at: Date.now() }
    : { available: false, at: Date.now() };
  if (result.available) {
    try { fs.mkdirSync(HOME, { recursive: true }); fs.writeFileSync(cacheFile, JSON.stringify(result), "utf8"); } catch { /* 忽略 */ }
  }
  return result;
}

function healthInfo() {
  return new Promise((resolve) => {
    const req = http.get(`${BASE_URL}/health`, { timeout: 3000 }, (res) => {
      let body = "";
      res.on("data", (chunk) => { if (body.length < 4096) body += chunk; });
      res.on("end", () => {
        let engine = null;
        try { engine = JSON.parse(body)?.engine ?? null; } catch { /* 非 JSON 健康响应 */ }
        resolve({ ok: res.statusCode === 200, engine });
      });
    });
    req.on("error", () => resolve({ ok: false, engine: null }));
    req.on("timeout", () => { req.destroy(); resolve({ ok: false, engine: null }); });
  });
}

async function health() { return (await healthInfo()).ok; }

// 依次执行命令并把 stdout/stderr 实时写入控制台与 install.log；非 0 退出码抛错。
function runLogged(cmd, args, options = {}) {
  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn(cmd, args, {
        cwd: fs.existsSync(HOME) ? HOME : os.homedir(),
        env: { ...process.env, ...(options.env || {}) },
        windowsHide: true,
        stdio: ["ignore", "pipe", "pipe"],
      });
    } catch (error) {
      reject(new Error(`无法启动 ${cmd}：${error?.message || error}`));
      return;
    }
    const pump = (stream, isError) => {
      stream.on("data", (chunk) => {
        const text = String(chunk);
        if (isError) process.stderr.write(text); else process.stdout.write(text);
        appendInstallLog(text);
      });
    };
    pump(child.stdout, false);
    pump(child.stderr, true);
    child.on("error", (error) => reject(new Error(`无法启动 ${cmd}：${error?.message || error}`)));
    child.on("close", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`命令失败（退出码 ${code}）：${path.basename(String(cmd))} ${args[0] === "-m" ? args.slice(0, 3).join(" ") : args[0] === "-c" ? "-c <python>" : args.slice(0, 3).join(" ")}`));
    });
  });
}

/* ---------------- 引擎安装状态 ---------------- */

const fileReady = (file, minBytes = 10 * 1024 * 1024) => {
  try { return fs.statSync(file).size > minBytes; } catch { return false; }
};

const whisperDeps = () => fs.existsSync(PY) && fs.existsSync(path.join(VENV, "Lib", "site-packages", "faster_whisper_server", "__main__.py"));
const whisperModelPath = () => {
  const modelDir = path.join(MODELS, HF_MODEL.split("/")[1]);
  const model = path.join(modelDir, "model.bin");
  return fileReady(model) ? model : null;
};
const whisperInstalled = () => whisperDeps() && Boolean(whisperModelPath());

const funasrDeps = () => fs.existsSync(FUNASR_FLAG);
const funasrModelPath = () => {
  for (const name of FUNASR_MODEL_FILES) {
    const file = path.join(FUNASR_MODEL_DIR, name);
    if (fileReady(file)) return file;
  }
  return null;
};
const funasrInstalled = () => funasrDeps() && Boolean(funasrModelPath());

/* ---------------- whisper 引擎 ---------------- */

function writeWhisperConfig() {
  const modelPath = whisperModelPath();
  if (!modelPath) throw new Error("faster-whisper 模型缺失，请重新执行「安装引擎」。");
  fs.mkdirSync(MODELS, { recursive: true });
  const yaml = [
    "batch_size: 1",
    "model_options:",
    "  device: cpu",
    "  compute_type: int8",
    "models:",
    `  - name: whisper-1`,
    `    path: ${path.dirname(modelPath).replace(/\\/g, "/")}`,
    "",
  ].join("\n");
  fs.writeFileSync(CONFIG_YAML, yaml, "utf8");
}

async function whisperInstall() {
  if (!whisperDeps()) {
    const python = findPython();
    if (!python) throw new Error("未找到 Python（需要 3.10-3.13）。请先安装 Python 并勾选 Add to PATH 后重试。");
    fs.mkdirSync(HOME, { recursive: true });
    if (!fs.existsSync(PY)) {
      setInstallPhase("venv", "1/3 创建 faster-whisper 虚拟环境…");
      await runLogged(python, ["-m", "venv", VENV]);
    }
    setInstallPhase("deps", "2/3 安装 faster-whisper-server（约 1-3 分钟）…");
    const pkgs = ["--no-cache-dir", "faster-whisper-server", "av==15.1.0"];
    try {
      await runLogged(PY, ["-m", "pip", "install", ...pkgs]);
    } catch (error) {
      appendInstallLog(`\n[local-asr] 依赖安装失败：${error.message}\n[local-asr] 切换清华 PyPI 镜像重试…\n`);
      await runLogged(PY, ["-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", ...pkgs]);
    }
  } else {
    appendInstallLog("[local-asr] faster-whisper 依赖已安装，跳过。\n");
  }
  if (!whisperModelPath()) {
    setInstallPhase("model", `3/3 下载模型 ${HF_MODEL}（约 460MB，视网速 5-20 分钟）…`);
    const modelDir = path.join(MODELS, HF_MODEL.split("/")[1]).replace(/\\/g, "/");
    const script = `from huggingface_hub import snapshot_download;` +
      `p=snapshot_download('${HF_MODEL}', local_dir=r'${modelDir}');` +
      `print('MODEL_DIR', p)`;
    try {
      await runLogged(PY, ["-c", script], { env: { HF_HUB_DISABLE_SYMLINKS_WARNING: "1" } });
    } catch (error) {
      appendInstallLog(`\n[local-asr] HuggingFace 下载失败：${error.message}\n[local-asr] 切换 hf-mirror.com 镜像重试…\n`);
      await runLogged(PY, ["-c", script], { env: { HF_HUB_DISABLE_SYMLINKS_WARNING: "1", HF_ENDPOINT: "https://hf-mirror.com" } });
    }
  } else {
    appendInstallLog("[local-asr] 模型已存在，跳过下载。\n");
  }
  setInstallPhase("finalize", "写入服务配置…");
  writeWhisperConfig();
  return { ok: true, engine: "whisper", home: HOME, model: HF_MODEL, baseUrl: BASE_URL };
}

async function whisperStart() {
  if (!whisperDeps()) throw new Error("faster-whisper 运行依赖未安装，请先执行「安装引擎」。");
  writeWhisperConfig();
  const logFd = fs.openSync(WHISPER_LOG, "a");
  const child = spawn(SERVER_CMD[0], [...SERVER_CMD.slice(1),
    "--host", "127.0.0.1", "--port", String(PORT), "--config", CONFIG_YAML,
  ], { cwd: HOME, detached: true, stdio: ["ignore", logFd, logFd], windowsHide: true, env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
  child.unref();
  fs.closeSync(logFd);
  fs.writeFileSync(WHISPER_PID_FILE, String(child.pid), "utf8");
  appendInstallLog(`[local-asr] faster-whisper 服务进程已启动（PID ${child.pid}），等待健康检查…\n`);
  let exited = null;
  child.once("exit", (code) => { exited = code; });
  for (let i = 0; i < 45; i++) {
    await sleep(1000);
    if (await health()) return { ok: true, engine: "whisper", pid: child.pid, baseUrl: BASE_URL };
    if (exited !== null) break;
  }
  const logTail = readLogTail(WHISPER_LOG, 4000);
  throw new Error(`faster-whisper 服务未能启动${exited !== null ? `（进程已退出，退出码 ${exited}）` : "（健康检查超时）"}。服务日志：\n${logTail || "(空)"}`);
}

/* ---------------- funasr 引擎（中文增强） ---------------- */

async function funasrInstall() {
  fs.mkdirSync(HOME, { recursive: true });
  const python = findPython();
  if (!python) throw new Error("未找到 Python（需要 3.9-3.13）。请先安装 Python 并勾选 Add to PATH 后重试。");
  if (!fs.existsSync(FUNASR_PY)) {
    setInstallPhase("venv", "1/3 创建 FunASR 虚拟环境…");
    await runLogged(python, ["-m", "venv", FUNASR_VENV]);
  }
  if (!funasrDeps()) {
    setInstallPhase("deps", "2/3 安装 funasr-onnx + modelscope（CPU 推理，无 torch，约 2-5 分钟）…");
    const pkgs = ["--no-cache-dir", "funasr-onnx", "modelscope", "numpy<2"];
    try {
      await runLogged(FUNASR_PY, ["-m", "pip", "install", ...pkgs]);
    } catch (error) {
      appendInstallLog(`\n[local-asr] 依赖安装失败：${error.message}\n[local-asr] 切换清华 PyPI 镜像重试…\n`);
      await runLogged(FUNASR_PY, ["-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", ...pkgs]);
    }
    fs.writeFileSync(FUNASR_FLAG, "funasr-onnx", "utf8");
  } else {
    appendInstallLog("[local-asr] funasr-onnx 依赖已安装，跳过。\n");
  }
  if (!funasrModelPath()) {
    setInstallPhase("model", "3/3 下载 Paraformer-large 中文模型（约 230MB，ModelScope 国内直连较快）…");
    const modelDir = FUNASR_MODEL_DIR.replace(/\\/g, "/");
    await runLogged(FUNASR_PY, ["-c",
      `from modelscope import snapshot_download;` +
      `p=snapshot_download('${FUNASR_MODEL_ID}', local_dir=r'${modelDir}');` +
      `print('MODEL_DIR', p)`]);
  } else {
    appendInstallLog("[local-asr] 模型已存在，跳过下载。\n");
  }
  setInstallPhase("finalize", "复制本地转写服务脚本…");
  fs.mkdirSync(path.dirname(FUNASR_SERVER_DST), { recursive: true });
  fs.copyFileSync(FUNASR_SERVER_SRC, FUNASR_SERVER_DST);
  return { ok: true, engine: "funasr", home: HOME, model: FUNASR_MODEL_ID, baseUrl: BASE_URL };
}

async function funasrStart() {
  if (!funasrDeps()) throw new Error("FunASR 运行依赖未安装，请先执行「安装引擎」。");
  if (!funasrModelPath()) throw new Error("FunASR 模型缺失，请重新执行「安装引擎」（已下载部分会自动续传）。");
  // 保持服务脚本为插件最新版
  fs.copyFileSync(FUNASR_SERVER_SRC, FUNASR_SERVER_DST);
  const logFd = fs.openSync(FUNASR_LOG, "a");
  const child = spawn(FUNASR_PY, [
    FUNASR_SERVER_DST,
    "--host", "127.0.0.1", "--port", String(PORT), "--model-dir", FUNASR_MODEL_DIR,
  ], { cwd: HOME, detached: true, stdio: ["ignore", logFd, logFd], windowsHide: true, env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
  child.unref();
  fs.closeSync(logFd);
  fs.writeFileSync(FUNASR_PID_FILE, String(child.pid), "utf8");
  appendInstallLog(`[local-asr] FunASR 服务进程已启动（PID ${child.pid}），等待健康检查…\n`);
  let exited = null;
  child.once("exit", (code) => { exited = code; });
  for (let i = 0; i < 45; i++) {
    await sleep(1000);
    if (await health()) return { ok: true, engine: "funasr", pid: child.pid, baseUrl: BASE_URL };
    if (exited !== null) break;
  }
  const logTail = readLogTail(FUNASR_LOG, 4000);
  throw new Error(`FunASR 服务未能启动${exited !== null ? `（进程已退出，退出码 ${exited}）` : "（健康检查超时）"}。服务日志：\n${logTail || "(空)"}`);
}

async function startEngine() {
  const info = await healthInfo();
  if (info.ok) {
    return { ok: true, alreadyRunning: true, engine: ENGINE, runningEngine: info.engine || engineFromPid(), baseUrl: BASE_URL };
  }
  return ENGINE === "funasr" ? await funasrStart() : await whisperStart();
}

function engineFromPid() {
  const readPid = (file) => {
    try { return Number(fs.readFileSync(file, "utf8").trim()) || 0; } catch { return 0; }
  };
  if (pidAlive(readPid(FUNASR_PID_FILE))) return "funasr";
  if (pidAlive(readPid(WHISPER_PID_FILE))) return "whisper";
  return null;
}

function stopByPidFile(file) {
  const readPid = () => {
    try { return Number(fs.readFileSync(file, "utf8").trim()) || 0; } catch { return 0; }
  };
  const pid = readPid();
  if (!pidAlive(pid)) return false;
  const r = spawnSync("taskkill", ["/F", "/PID", String(pid)], { encoding: "utf8", windowsHide: true });
  return r.status === 0;
}

// 兜底：按监听端口找 PID 杀掉（pid 文件丢失/服务被手动重启过时仍能停干净）
function killPortListener(port) {
  const r = spawnSync("netstat", ["-ano"], { encoding: "utf8", windowsHide: true });
  const pids = new Set(String(r.stdout || "").split("\n")
    .filter((line) => line.includes(`:${port} `) && /LISTENING/i.test(line))
    .map((line) => line.trim().split(/\s+/).pop())
    .filter((pid) => /^\d+$/.test(pid) && Number(pid) !== process.pid));
  let killed = false;
  for (const pid of pids) {
    const k = spawnSync("taskkill", ["/F", "/PID", String(pid)], { encoding: "utf8", windowsHide: true });
    if (k.status === 0) killed = true;
  }
  return killed;
}

/* ---------------- 命令实现 ---------------- */

async function cmdStatus() {
  const runningInfo = await healthInfo();
  const install = installSnapshot();
  const engines = {
    whisper: {
      label: engineLabel("whisper"),
      installed: whisperInstalled(),
      depsInstalled: whisperDeps(),
      modelReady: Boolean(whisperModelPath()),
      model: HF_MODEL.split("/")[1],
    },
    funasr: {
      label: engineLabel("funasr"),
      installed: funasrInstalled(),
      depsInstalled: funasrDeps(),
      modelReady: Boolean(funasrModelPath()),
      model: "paraformer-large-onnx",
    },
  };
  const selected = engines[ENGINE];
  const installed = selected.installed;
  const runningEngine = runningInfo.ok ? (runningInfo.engine || engineFromPid()) : null;
  const python = selected.depsInstalled ? null : probePython();
  const model = ENGINE === "funasr"
    ? "FunASR Paraformer-large（中文优化 · onnx CPU）"
    : `${HF_MODEL}（faster-whisper）`;

  let next;
  if (install.running) {
    next = `正在安装（${install.message || install.phase || "处理中"}），进度与日志会自动刷新。`;
  } else if (runningInfo.ok) {
    next = runningEngine && runningEngine !== ENGINE
      ? `端口 ${PORT} 上正在运行 ${engineLabel(runningEngine)}，与当前选择的 ${engineLabel()} 不同；如需切换请先「停止服务」再「启动服务」。`
      : `服务已就绪。把会议助手的转写通道设为 local，服务地址填 ${BASE_URL}/v1，本地模型名 whisper-1。`;
  } else if (install.stale) {
    next = `上次安装没有完成（${install.error || install.message || "进程已退出"}）。可以重新点击「安装引擎」续装，详细日志见下方。`;
  } else if (!selected.depsInstalled) {
    next = `引擎（${engineLabel()}）未安装，请点击「安装引擎」。`;
  } else if (!selected.modelReady) {
    next = `运行依赖已装好，但模型还没下载完成；再次点击「安装引擎」会续传模型。`;
  } else {
    next = `引擎（${engineLabel()}）已安装但服务未运行，请点击「启动服务」。`;
  }

  out({
    engine: ENGINE,
    engineLabel: engineLabel(),
    installed,
    running: runningInfo.ok,
    runningEngine,
    baseUrl: runningInfo.ok ? BASE_URL : null,
    model,
    home: HOME,
    venv: ENGINE === "funasr" ? FUNASR_VENV : VENV,
    port: PORT,
    engines,
    python,
    install,
    logFile: INSTALL_LOG,
    next,
  });
}

async function cmdInstall() {
  fs.mkdirSync(HOME, { recursive: true });
  resetInstallLog();
  appendInstallLog(`[local-asr] ${new Date().toLocaleString()} · 引擎 ${engineLabel()} · 安装目录 ${HOME}\n`);
  appendInstallLog(`[local-asr] 安装目录: ${HOME} · 引擎: ${engineLabel()}\n`);
  writeInstallState({
    running: true,
    pid: process.pid,
    engine: ENGINE,
    phase: "prepare",
    message: "检查运行环境…",
    startedAt: new Date().toISOString(),
    finishedAt: null,
    ok: null,
    error: "",
    stale: false,
  });
  try {
    const data = ENGINE === "funasr" ? await funasrInstall() : await whisperInstall();
    writeInstallState({
      running: false,
      phase: "done",
      message: "安装完成",
      ok: true,
      error: "",
      finishedAt: new Date().toISOString(),
    });
    appendInstallLog(`\n[local-asr] 安装完成：${HOME}\n`);
    out({
      ...data,
      hint: `安装完成。执行「启动服务」后，在会议助手中把转写通道设为 local，服务地址 ${BASE_URL}/v1，模型名 whisper-1。`,
    });
  } catch (error) {
    const message = String(error?.message || error);
    const previous = readInstallState();
    const failedPhase = previous?.phase && previous.phase !== "error" ? previous.phase : previous?.failedPhase;
    writeInstallState({
      running: false,
      phase: "error",
      failedPhase,
      message: "安装失败",
      ok: false,
      error: message,
      finishedAt: new Date().toISOString(),
    });
    appendInstallLog(`\n[local-asr] 安装失败：${message}\n`);
    throw error;
  }
}

async function cmdStart() {
  appendInstallLog(`[local-asr] 启动引擎: ${engineLabel()}\n`);
  const data = await startEngine();
  if (data.alreadyRunning) {
    out({ ...data, hint: `服务已在运行：${BASE_URL}/v1，模型名 whisper-1。` });
    return;
  }
  appendInstallLog(`[local-asr] 服务就绪：${BASE_URL}/v1\n`);
  out({ ...data, hint: data.engine === "funasr"
    ? `FunASR 服务就绪：${BASE_URL}/v1，模型名 whisper-1。首次转写会加载模型（约 10-30 秒）。`
    : `faster-whisper 服务就绪：${BASE_URL}/v1，模型名 whisper-1。` });
}

function cmdStop() {
  const runningEngine = engineFromPid();
  const killedFunasr = stopByPidFile(FUNASR_PID_FILE);
  const killedWhisper = stopByPidFile(WHISPER_PID_FILE);
  // 兜底：两个 pid 文件都没命中时，按端口找监听进程杀掉
  const killedByPort = !killedFunasr && !killedWhisper ? killPortListener(PORT) : false;
  if (killedFunasr || killedWhisper || killedByPort) {
    const label = runningEngine ? engineLabel(runningEngine) : "本地转写";
    out({ ok: true, killed: true, note: `${label}服务已停止。` });
  } else {
    out({ ok: true, killed: false, note: "没有正在运行的服务进程。" });
  }
}

function writeToneWav(file) {
  const sampleRate = 16000;
  const seconds = 2;
  const total = sampleRate * seconds;
  const data = Buffer.alloc(total * 2);
  for (let i = 0; i < total; i++) data.writeInt16LE(Math.round(8000 * Math.sin(2 * Math.PI * 440 * i / sampleRate)), i * 2);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(1, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, Buffer.concat([header, data]));
}

function postTranscription(filePath, timeoutMs = 180000) {
  return new Promise((resolve) => {
    const boundary = `----ruodao${Date.now()}`;
    const audio = fs.readFileSync(filePath);
    const head = Buffer.from(
      `--${boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n` +
      `--${boundary}\r\nContent-Disposition: form-data; name="file"; filename="asr-test.wav"\r\nContent-Type: audio/wav\r\n\r\n`,
    );
    const tail = Buffer.from(`\r\n--${boundary}--\r\n`);
    const body = Buffer.concat([head, audio, tail]);
    const req = http.request(`${BASE_URL}/v1/audio/transcriptions`, {
      method: "POST",
      timeout: timeoutMs,
      headers: {
        "Content-Type": `multipart/form-data; boundary=${boundary}`,
        "Content-Length": body.length,
      },
    }, (res) => {
      let text = "";
      res.on("data", (chunk) => { text += chunk; });
      res.on("end", () => resolve({ status: res.statusCode, body: text }));
    });
    req.on("error", (error) => resolve({ status: 0, body: String(error?.message || error) }));
    req.on("timeout", () => { req.destroy(); resolve({ status: 0, body: `请求超时（${Math.round(timeoutMs / 1000)}s）` }); });
    req.write(body);
    req.end();
  });
}

async function cmdTest() {
  if (!(await health())) throw new Error("服务未运行，请先执行「启动服务」。");
  const wav = path.join(HOME, "asr-test.wav");
  writeToneWav(wav);
  const result = await postTranscription(wav);
  let parsed = null;
  try { parsed = JSON.parse(result.body); } catch { /* 保留原文 */ }
  const ok = result.status === 200 && parsed && typeof parsed.text === "string";
  out({
    ok,
    response: result.body,
    note: ok
      ? `接口可用（引擎 ${engineLabel()}）。测试音频为提示音，仅验证链路；建议在会议助手里用真实录音再验证一次。`
      : `转写自检失败（HTTP ${result.status}）。`,
  });
}

/** 一键配置会议助手：探测设备 + 写入 local 转写通道全部配置 */
function ffmpegAudioDevices() {
  const r = spawnSync("ffmpeg", ["-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
    { encoding: "utf8", windowsHide: true });
  return [...String(r.stderr || "").matchAll(/"([^"]+)" \(audio\)/g)].map((m) => m[1]);
}

async function cmdApplyConfig() {
  const channel = (process.argv[3] || "local").toLowerCase(); // local | cloud
  fs.mkdirSync(HOME, { recursive: true });
  const appData = process.env.APPDATA || path.join(os.homedir(), "AppData", "Roaming");
  const dataDir = process.env.RUODAO_PLUGIN_DATA || path.join(appData, "Ruodao Agent", "plugin-data");
  const configFile = path.join(dataDir, "meeting-assistant.json");

  // 1. 确保本地服务在运行（仅 local 通道需要）——走 startEngine 内联，避免重复输出 JSON
  if (channel === "local") {
    if (ENGINE === "funasr" && !funasrDeps()) throw new Error("FunASR 引擎未安装，请先执行「安装引擎」。");
    if (ENGINE === "whisper" && !whisperDeps()) throw new Error("引擎未安装，请先执行「安装引擎」。");
    if (ENGINE === "funasr" && !funasrModelPath()) throw new Error("FunASR 模型缺失，请先执行「安装引擎」。");
    if (ENGINE === "whisper" && !whisperModelPath()) throw new Error("faster-whisper 模型缺失，请先执行「安装引擎」。");
    if (!(await health())) await startEngine();
  }

  // 2. 探测音频设备
  const devices = ffmpegAudioDevices();
  const mic = devices.find((d) => /麦克风|microphone/i.test(d)) || devices[0] || "";
  const loopback = devices.find((d) => /virtual-audio-capturer|立体声混音|stereo mix/i.test(d)) || "";

  // 3. 探测本机 Ollama：在线则大脑走 local，否则保持宿主绑定（host）
  const ollamaAlive = await new Promise((resolve) => {
    const req = http.get("http://127.0.0.1:11434/api/tags", { timeout: 1500 }, (res) => {
      res.resume();
      resolve(res.statusCode === 200);
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });

  // 4. 写入会议助手配置（保留用户已填字段，仅补空）
  let data = { config: {} };
  if (fs.existsSync(configFile)) {
    try { data = JSON.parse(fs.readFileSync(configFile, "utf8")); } catch { /* 重建 */ }
  }
  if (!data.config || typeof data.config !== "object") data.config = {};
  const c = data.config;
  const put = (k, v) => { if (!c[k]) c[k] = v; };
  if (channel === "local") {
    // 新版统一选择器 + 旧字段兼容（旧版会议助手仍读 CHANNEL/BASE_URL/MODEL）
    c.RUODAO_STT_SERVICE = "local";
    c.RUODAO_STT_CHANNEL = "local";
    c.RUODAO_LOCAL_STT_BASE_URL = BASE_URL + "/v1";
    c.RUODAO_LOCAL_STT_MODEL = "whisper-1";
  } else {
    c.RUODAO_STT_SERVICE = "host";
    c.RUODAO_STT_CHANNEL = "cloud";
  }
  if (ollamaAlive) {
    c.RUODAO_LLM_CHANNEL = "local";
    put("RUODAO_LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434/v1");
    put("RUODAO_LOCAL_LLM_MODEL", "qwen2.5:7b");
  } else {
    put("RUODAO_LLM_CHANNEL", "host");
  }
  put("RUODAO_AUDIO_DEVICE", mic);
  put("RUODAO_SYSTEM_AUDIO_DEVICE", loopback);
  put("RUODAO_CAPTURE_SECONDS", "12");
  put("RUODAO_MEETING_ROLE", "资深会议参谋");
  fs.mkdirSync(dataDir, { recursive: true });
  fs.writeFileSync(configFile, JSON.stringify(data, null, 2), "utf8");

  out({
    ok: true,
    engine: ENGINE,
    channel,
    configFile,
    brain: ollamaAlive ? "local（检测到本机 Ollama，已自动切到本地模型）" : "host（未检测到 Ollama，保持若道模型绑定；如需云端大脑请在会议助手设置把 AI 大脑改为 cloud 并填 Key）",
    applied: c,
    detected: { audioDevices: devices, mic, loopback, loopbackNote: loopback ? "" : "未检测到系统内录设备（virtual-audio-capturer / 立体声混音），线上会议对方声音通道暂不可用，麦克风通道不受影响。" },
    hint: `配置已写入（本地引擎：${engineLabel()}）。请关闭并重新打开「会议助手」工作台页面即可看到已填好的配置；若当前页已打开，点一次「保存配置」也不会覆盖（字段已就绪）。`,
  });
}

/* ---------------- 入口 ---------------- */

const cmd = process.argv[2];
const handlers = { status: cmdStatus, install: cmdInstall, start: cmdStart, stop: cmdStop, test: cmdTest, "apply-config": cmdApplyConfig };
if (!handlers[cmd]) {
  console.error("用法: node local-asr.mjs <install|start|stop|status|test|apply-config [local|cloud]>（引擎由 RUODAO_ASR_ENGINE=whisper|funasr 决定）");
  process.exit(2);
}

try {
  applyPluginConfig(await readPluginConfig());
  refreshPaths();
  await handlers[cmd]();
} catch (error) {
  const message = String(error?.message || error);
  if (cmd === "install") {
    const previous = readInstallState();
    const failedPhase = previous?.phase && previous.phase !== "error" ? previous.phase : previous?.failedPhase;
    writeInstallState({ running: false, phase: "error", failedPhase, message: "安装失败", ok: false, error: message, finishedAt: new Date().toISOString() });
    appendInstallLog(`\n[local-asr] 安装失败：${message}\n`);
  }
  console.error(`[local-asr] ${message}`);
  out({ ok: false, error: message, logFile: INSTALL_LOG || "" });
  process.exitCode = 1;
}
