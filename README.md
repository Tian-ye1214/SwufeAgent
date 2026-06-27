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
pip install redlotus                 # 核心
pip install "redlotus[browser]"      # 浏览器 Skill（装后再 playwright install chromium）
pip install "redlotus[bots]"         # QQ / 微信 机器人
```

首次运行会在**用户配置目录**生成 `config.json`（Windows：`%LOCALAPPDATA%\RedLotus`；Linux：`~/.config/RedLotus`）。在其中填好 `BASE_URL`、`API_KEY`、`RAG_models` 与各角色模型；也可用环境变量或就近的 `.env` 覆盖同名项。日志 / 向量库（RAG）/ 长期记忆落在**用户数据目录**；Agent 工作产物 `WorkDatabase/` 与会话快照 `.redlotus/` 落在你**启动时的当前目录**。

```bash
redlotus                        # 终端见真章
```

想让它在群里说话：

```bash
python -m redlotus.API.QQ       # 需配置 QQBOT 等环境变量
python -m redlotus.API.WeChat   # 扫码登录
```

QQ 机器人前置：先装好并运行 [NapCat](https://github.com/NapNeko/NapCatQQ)（提供 OneBot WebSocket 接口）。首次会在用户配置目录从随包模板生成 `config.yaml`，在其中填好 `bt_uin`（机器人 QQ 号，也可用环境变量 `QQBOT_ID` 覆盖）与 `napcat.webui_token`（需为强密码：至少 12 位且含数字、大小写字母与特殊符号）。配置缺失或无效时启动会直接报错退出，不会静默卡住。

---

## 给折腾的人

```bash
pip install -e ".[dev]"         # 开发安装（仓库根；之后 redlotus 命令直连源码）
python main.py                  # 等价于 redlotus，便于断点调试
pytest tests/ -q                # 冒烟
pip install ".[build]" && pyinstaller build.spec   # 打成可执行目录（可选）
```

代码在 `src/redlotus/`，命令入口 `redlotus`（= `redlotus.agent_core.entrypoint:main`）。随包默认配置 `src/redlotus/config.json` 仅作首次 seed 用，运行时实际读写**用户配置目录**里的 `config.json`。

---

*启动后你会看到那行字：* **❦ ────  红莲极意  ·  RedLotus Agent  ──── ❦**
