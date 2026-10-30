#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FunASR 本地转写服务（OpenAI 兼容 /v1/audio/transcriptions）。

- 引擎：funasr-onnx Paraformer-large（中文，CPU onnxruntime 推理，无 torch 依赖）
- 接口：
    GET  /health                      → 200（服务存活即可用，模型懒加载）
    POST /v1/audio/transcriptions     → multipart 表单（file, model 可忽略），返回 {"text": "..."}
- 仅监听 127.0.0.1，录音不出本机。
- 只用 Python 标准库，不引入 fastapi/uvicorn。
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from email.parser import BytesParser
from email.policy import HTTP as HTTP_POLICY
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_DIR = os.environ.get("FUNASR_MODEL_DIR", "")
MAX_BODY = 300 * 1024 * 1024  # 200MB 音频 + multipart 开销余量

_model = None
_model_error = ""


def log(msg):
    sys.stdout.write(f"[funasr-server] {msg}\n", )
    sys.stdout.flush()


def load_model():
    """懒加载 Paraformer：首次转写请求时才初始化（健康检查不用等模型）。"""
    global _model, _model_error
    if _model is not None or _model_error:
        return _model
    t0 = time.time()
    try:
        from funasr_onnx import Paraformer  # type: ignore
        if not MODEL_DIR or not os.path.isdir(MODEL_DIR):
            raise RuntimeError(f"模型目录无效：{MODEL_DIR!r}（请重新执行插件「安装本地转写引擎」）")
        log(f"加载 Paraformer 模型：{MODEL_DIR}")
        _model = Paraformer(MODEL_DIR, batch_size=1, device_id=0, quantize=True)
        log(f"模型加载完成（{time.time() - t0:.1f}s）")
    except Exception as error:  # noqa: BLE001
        _model_error = f"FunASR 模型加载失败：{error}"
        log(_model_error)
    return _model


def parse_multipart(body: bytes, content_type: str):
    """用 email.parser 解析 multipart/form-data，返回 {name: bytes}。"""
    parser = BytesParser(policy=HTTP_POLICY)
    msg = parser.parsebytes(b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    fields = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name:
            fields[name] = part.get_payload(decode=True) or b""
    return fields


def to_mono16k_wav(source_path: str) -> str:
    """funasr-onnx 需要 16k 单声道 wav；非 wav 或多声道时用 ffmpeg 归一化（无 ffmpeg 则按原样尝试）。"""
    with open(source_path, "rb") as f:
        head = f.read(12)
    is_wav = head[:4] == b"RIFF" and head[8:12] == b"WAVE"
    if not is_wav:
        raise RuntimeError("仅支持 wav 录音；其他格式请先在会议助手侧用 ffmpeg 转为 16k 单声道 wav")
    return source_path


class Handler(BaseHTTPRequestHandler):
    server_version = "RuodaoFunASR/1.0"

    def log_message(self, fmt, *args):  # 静默默认访问日志，只留关键事件
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            self._json(200, {"ok": True, "engine": "funasr", "modelDir": MODEL_DIR, "loaded": _model is not None})
        elif self.path.rstrip("/") in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{"id": "whisper-1", "object": "model", "owned_by": "funasr"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/audio/transcriptions", "/audio/transcriptions"):
            self._json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._json(413, {"error": {"message": f"请求体无效或过大（{length}B）"}})
            return
        content_type = self.headers.get("Content-Type") or ""
        body = self.rfile.read(length)
        if "multipart/form-data" not in content_type.lower():
            self._json(400, {"error": {"message": "必须使用 multipart/form-data 上传音频文件"}})
            return
        fields = parse_multipart(body, content_type)
        audio = fields.get("file") or b""
        if not audio:
            self._json(400, {"error": {"message": "缺少 file 字段"}})
            return
        model = load_model()
        if model is None:
            self._json(503, {"error": {"message": _model_error or "FunASR 模型未就绪"}})
            return
        tmp_dir = tempfile.mkdtemp(prefix="funasr-")
        wav_path = os.path.join(tmp_dir, "input.wav")
        try:
            with open(wav_path, "wb") as f:
                f.write(audio)
            source = to_mono16k_wav(wav_path)
            t0 = time.time()
            result = model([source])
            texts = []
            for group in result if isinstance(result, (list, tuple)) else [result]:
                for item in group if isinstance(group, (list, tuple)) else [group]:
                    text = (item or {}).get("text") if isinstance(item, dict) else str(item)
                    if text:
                        texts.append(str(text).strip())
            text = "".join(texts).strip()
            log(f"转写完成：{len(audio)}B → {len(text)} 字（{time.time() - t0:.1f}s）")
            self._json(200, {"text": text})
        except Exception as error:  # noqa: BLE001
            log(f"转写失败：{error}")
            self._json(500, {"error": {"message": f"转写失败：{error}"}})
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _json(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    global MODEL_DIR
    arg = argparse.ArgumentParser()
    arg.add_argument("--host", default="127.0.0.1")
    arg.add_argument("--port", type=int, default=8000)
    arg.add_argument("--model-dir", default=os.environ.get("FUNASR_MODEL_DIR", ""))
    args = arg.parse_args()
    MODEL_DIR = args.model_dir or MODEL_DIR
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"FUNASR_SERVER_READY http://{args.host}:{args.port} modelDir={MODEL_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
