# GLM-OCR Service

基于 GLM-OCR + mlx-vlm 的本地 OCR 服务，针对 Apple Silicon 优化。

## 快速部署

```bash
git clone https://github.com/yeuxuan/ocr-service.git
cd ocr-service
./setup.sh        # 一键安装环境 + 下载模型
./ocr-service.sh start   # 启动服务
```

### 服务管理

```bash
./ocr-service.sh start    # 启动
./ocr-service.sh stop     # 停止
./ocr-service.sh restart  # 重启
./ocr-service.sh status   # 查看状态
./ocr-service.sh logs     # 实时日志 (可选 mlx / api / all)
```

---

## API 文档

**Base URL**: `http://127.0.0.1:5002`

交互式文档: `http://127.0.0.1:5002/docs`

---

### 1. 提交 OCR 任务

上传图片或 PDF 文件，创建异步 OCR 任务。

```
POST /api/tasks
Content-Type: multipart/form-data
```

**请求参数**

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | file | 是 | 图片或 PDF，最大 100MB |

**支持格式**: `.png` `.jpg` `.jpeg` `.webp` `.bmp` `.tiff` `.tif` `.pdf`

**响应** `200`

```json
{
  "task_id": "aaef1519ad1741228f6f50d0093f853a",
  "status": "pending",
  "message": "Subscribe to GET /api/tasks/{task_id}/events for real-time updates"
}
```

**错误码**

| 状态码 | 说明 |
|--------|------|
| 400 | 未提供文件 / 不支持的文件类型 |
| 413 | 文件超过 100MB |

**示例**

```bash
curl -X POST http://127.0.0.1:5002/api/tasks \
  -F "file=@/path/to/document.png"
```

---

### 2. SSE 实时订阅任务进度

通过 Server-Sent Events 实时接收任务状态变更，适用于需要等待结果的场景。

```
GET /api/tasks/{task_id}/events
```

**响应**: `text/event-stream`

连接后会立即推送当前状态，之后每次状态变更时推送。无变更时每 30 秒发送心跳（保持 Cloudflare Tunnel 连接）。任务完成或失败后连接自动关闭。

**事件类型**

| event | 说明 | 触发时机 |
|-------|------|----------|
| `pending` | 排队中 | 连接时任务尚未处理 |
| `processing` | 处理中 | Worker 开始处理 |
| `completed` | 完成 | OCR 识别成功，data 包含结果 |
| `failed` | 失败 | 处理出错，data 包含 error |
| `heartbeat` | 心跳 | 30 秒无状态变更 |

**事件格式**

```
event: processing
data: {"task_id": "aaef1519...", "status": "processing", "file_name": "doc.png", ...}

event: completed
data: {"task_id": "aaef1519...", "status": "completed", "result": {"markdown": "...", "json": {...}}, ...}
```

**completed 事件 data.result 结构**

```json
{
  "markdown": "识别出的完整 Markdown 文本",
  "json": {
    "pages": [
      {
        "page_idx": 0,
        "blocks": [
          {
            "type": "text",
            "bbox": [x1, y1, x2, y2],
            "content": "识别出的文字内容"
          }
        ]
      }
    ]
  },
  "file_name": "document.png"
}
```

**示例**

```bash
curl -N http://127.0.0.1:5002/api/tasks/aaef1519ad1741228f6f50d0093f853a/events
```

**JavaScript 示例**

```javascript
const es = new EventSource('/api/tasks/aaef1519.../events');

es.addEventListener('completed', (e) => {
  const data = JSON.parse(e.data);
  console.log('OCR 结果:', data.result.markdown);
  es.close();
});

es.addEventListener('failed', (e) => {
  const data = JSON.parse(e.data);
  console.error('失败:', data.error);
  es.close();
});
```

---

### 3. 查询任务状态

轮询方式获取任务状态和结果。

```
GET /api/tasks/{task_id}
```

**响应** `200`

```json
{
  "task_id": "aaef1519ad1741228f6f50d0093f853a",
  "status": "completed",
  "created_at": "2026-05-13T14:30:00+00:00",
  "completed_at": "2026-05-13T14:30:15+00:00",
  "file_name": "document.png",
  "result": {
    "markdown": "...",
    "json": {...},
    "file_name": "document.png"
  },
  "error": null
}
```

| `status` 值 | 说明 |
|-------------|------|
| `pending` | 排队等待处理 |
| `processing` | 正在 OCR 识别 |
| `completed` | 完成，`result` 字段有结果 |
| `failed` | 失败，`error` 字段有原因 |

**错误码**: `404` — 任务不存在

---

### 4. 列出任务

```
GET /api/tasks
```

**查询参数**

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `limit` | int | 20 | 返回数量，1-200 |
| `status` | string | 无 | 按状态过滤：`pending` / `processing` / `completed` / `failed` |

**响应** `200`

```json
[
  {
    "task_id": "aaef1519...",
    "status": "completed",
    "file_name": "document.png",
    "created_at": "2026-05-13T14:30:00+00:00",
    "completed_at": "2026-05-13T14:30:15+00:00",
    "error": null
  }
]
```

**示例**

```bash
# 查看最近 10 个已完成的任务
curl "http://127.0.0.1:5002/api/tasks?status=completed&limit=10"
```

---

### 5. 健康检查

```
GET /health
```

**响应** `200`

```json
{"status": "ok"}
```

---

## 典型调用流程

```
1. POST /api/tasks          ← 上传文件，拿到 task_id
2. GET  /api/tasks/{id}/events  ← SSE 订阅，等 completed 事件
3. 从 completed 事件的 data.result 中提取 markdown / json 结果
```

```bash
# 一行搞定：上传 + 等待结果
TASK_ID=$(curl -s -F "file=@doc.png" http://127.0.0.1:5002/api/tasks | jq -r .task_id)
curl -N "http://127.0.0.1:5002/api/tasks/${TASK_ID}/events"
```
