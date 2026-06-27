# ❦ 红莲极意 · RedLotus

[![PyPI](https://img.shields.io/pypi/v/redlotus.svg)](https://pypi.org/project/redlotus/)
[![Python](https://img.shields.io/pypi/pyversions/redlotus.svg)](https://pypi.org/project/redlotus/)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

> 一个好的 Agent 项目，从来不需要过度设计。  
> 但它需要记得你说过的话、会叫帮手、还能在 QQ 里回你。

**红莲**是一朵会自己长技能的 AI：终端里和你对话，复杂事丢给 Worker，乱局交给 Coordinator 收拾；聊过的写进短时记忆，重要的烙进长期记忆；内置文件提取能读 PDF 的文本、表格、图片和页面结构，量化、回测、爬网页等领域能力则靠技能（skills）现学现用，而不是堆一层又一层的框架。

```bash
pip install redlotus && redlotus
```

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
- **可武装**：技能即插即用，Agentic Coding、行情、回测开箱即用，还能用 `clawhub` 现装现用

底层：[Pydantic AI](https://ai.pydantic.dev/) · LanceDB · 你配置的任意 OpenAI 兼容 API

---

## 点火

**Python 3.12+** · Windows / Linux / macOS 均可

最直接——装上就能用：

```bash
pip install redlotus
redlotus
```

> `redlotus` 命令会装进你当前 Python 的 `Scripts`（Win）/ `bin`（Linux·mac）目录。只要该目录在 PATH 上（python.org、conda 默认都在），**之后随便开个终端敲 `redlotus` 就启动**。

想要一个**独立隔离、不污染你 Python 环境**的全局命令（推荐纯命令行使用）：

```bash
uv tool install redlotus      # 或者：pipx install redlotus
redlotus
```

这会单独建一个隔离环境，并把 `redlotus` 挂到全局 PATH——任意新终端直接启动，且与你别的项目互不干扰。

**可选能力（extras，按需装）：**

```bash
pip install "redlotus[browser]"   # 浏览器 Skill（装后再 playwright install chromium）
pip install "redlotus[bots]"      # QQ / 微信 机器人
pip install "redlotus[viz]"       # 绘图 / PDF 生成 / 图像（量化回测等技能用）
pip install "redlotus[all]"       # 一次装全
```
> extras 同样适用于 `uv tool install "redlotus[browser]"` 这种写法。

### 配置与数据落点

首次运行会在**用户配置目录**生成 `config.json`（Windows：`%LOCALAPPDATA%\RedLotus`；Linux：`~/.config/RedLotus`；macOS：`~/Library/Application Support/RedLotus`）。在其中填好 `BASE_URL`、`API_KEY`、`RAG_models` 与各角色模型；也可用**环境变量**或就近的 `.env` 覆盖同名项（优先级：环境变量 → `.env` → `config.json`）。

- **跟随用户（全局）**：日志、向量库（RAG）、长期记忆，落在用户数据目录，换工作目录也不丢。
- **跟随工作目录**：Agent 工作产物 `WorkDatabase/` 与会话快照 `.redlotus/`，落在你**启动时的当前目录**。

---

## 让它在群里说话（可选）

```bash
pip install "redlotus[bots]"
python -m redlotus.API.QQ       # 需配置 QQBOT 等环境变量
python -m redlotus.API.WeChat   # 扫码登录
```

QQ 机器人前置：先装好并运行 [NapCat](https://github.com/NapNeko/NapCatQQ)（提供 OneBot WebSocket 接口）。首次会在用户配置目录从随包模板生成 `config.yaml`，在其中填好 `bt_uin`（机器人 QQ 号，也可用环境变量 `QQBOT_ID` 覆盖）与 `napcat.webui_token`（需为强密码：至少 12 位且含数字、大小写字母与特殊符号）。配置缺失或无效时启动会直接报错退出，不会静默卡住。

---

## 给折腾的人

```bash
git clone <repo> && cd Agent
pip install -e ".[dev]"         # 开发安装（仓库根；之后 redlotus 命令直连源码）
python main.py                  # 等价于 redlotus，便于断点调试
pytest tests/ -q                # 冒烟
pip install ".[build]" && pyinstaller build.spec   # 打成可执行目录（可选）
```

代码在 `src/redlotus/`，命令入口 `redlotus`（= `redlotus.agent_core.entrypoint:main`）。随包默认配置 `src/redlotus/config.json` 仅作首次 seed 用，运行时实际读写**用户配置目录**里的 `config.json`。

**自己发版**（构建 + 上传 PyPI，token 形式）：

```bash
uv build                                    # 产出 dist/*.whl 与 *.tar.gz
uv publish --token pypi-<你的-pypi-token>   # 或先设 UV_PUBLISH_TOKEN 再 uv publish
```

---

*启动后你会看到那行字：* **❦ ────  红莲极意  ·  RedLotus Agent  ──── ❦**
