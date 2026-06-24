# ❦ 红莲极意 · RedLotus

> 一个好的 Agent 项目，从来不需要过度设计。  
> 但它需要记得你说过的话、会叫帮手、还能在 QQ 里回你。

**红莲**是一朵会自己长技能的 AI：终端里和你对话，复杂事丢给 Worker，乱局交给 Coordinator 收拾；聊过的写进短时记忆，重要的烙进长期记忆；内置文件提取能读 PDF 的文本、表格、图片和页面结构，量化、回测、爬网页等领域能力则靠 `skills/` 现学现用，而不是堆一层又一层的框架。

---

## 它大概长什么样

```
你 ──► Manager（统筹）──► Worker × N（干活）
              │
              └── Coordinator（收尾、对齐上下文）
```

- **记得住**：向量检索 + 长短时记忆，不是每次开机都失忆
- **会分工**：简单自己答，难了派工，不硬扛
- **能出门**：CLI 是主场，QQ / 微信机器人是副线（可选）
- **可武装**：`src/skills/` 即插即用，Agentic Coding、行情、回测都在里面

底层：[Pydantic AI](https://ai.pydantic.dev/) · LanceDB · 你配置的任意 OpenAI 兼容 API

---

## 点火

**Python 3.12+** · Windows / Linux / macOS 均可

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -U pip && pip install -e ".[dev]"
playwright install chromium     # 要用浏览器 Skill 时再装
```

在 `src/config.json` 里填好 `BASE_URL`、`API_KEY` 和各角色模型；或在项目根放 `.env` 覆盖同名项。RAG 用 `RAG_models`，数据默认落在 `data/`。

```bash
python main.py                  # 终端见真章
```

想让它在群里说话：

```bash
cd src && python -m API.QQ      # 需配置 QQBOT 等环境变量
cd src && python -m API.WeChat   # 扫码登录
```

QQ 机器人前置：先装好并运行 [NapCat](https://github.com/NapNeko/NapCatQQ)（提供 OneBot WebSocket 接口）。配置文件 `config.yaml` 须放在 `src/API/` 下（已随仓库附带模板），其中填好 `bt_uin`（机器人 QQ 号，也可用环境变量 `QQBOT_ID` 覆盖）与 `napcat.webui_token`（需为强密码：至少 12 位且含数字、大小写字母与特殊符号）。配置缺失或无效时启动会直接报错退出，不会静默卡住。

---

## 给折腾的人

```bash
pytest tests/ -q                # 冒烟
pyinstaller build.spec          # 打成可执行目录（可选）
```

代码在 `src/`，入口 `main.py`，配置 `src/config.json`。

---

*启动后你会看到那行字：* **❦ ────  红莲极意  ·  RedLotus Agent  ──── ❦**
