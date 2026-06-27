## Skills Directory Description

- All Skills are located in the **`skills/` folder ({skills_root_path}) under the project root directory.

- **Stored by Skill directory:** Each **subfolder** under `skills/` corresponds to one Skill; the subfolder name can differ from the `name` field in the YAML file `SKILL.md`, but the system uses `name` for registration.

- Each Skill directory must contain a **`SKILL.md`: The file begins with YAML metadata (including `name`, `description`, etc.), followed by the main instructions.

- Reference materials, rules, scripts, etc. (such as `references/`, `rules/`) can be placed in the same Skill directory, and loaded as needed using tools.

- The system will rescan after file changes (including hot reloading); the following "Available Skills" list will be updated with the scan results.