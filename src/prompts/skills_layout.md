## Skills 目录说明

- 所有 Skills 位于项目根目录下的 **`skills/`** 文件夹（{skills_root_path}）。
- **按 Skill 分目录存放**：`skills/` 下每个**子文件夹**对应一个 Skill；子文件夹名可与 `SKILL.md` 里 YAML 的 `name` 字段不同，系统以 `name` 为准注册。
- 每个 Skill 目录内须有 **`SKILL.md`**：文件开头为 YAML 前置元数据（含 `name`、`description` 等），其后为正文指令。
- 同一 Skill 目录下可放参考资料、规则、脚本等（如 `references/`、`rules/`），按需通过工具加载。
- 变更文件后系统会重新扫描（含热加载）；下列「可用 Skills」列表会随扫描结果更新。
