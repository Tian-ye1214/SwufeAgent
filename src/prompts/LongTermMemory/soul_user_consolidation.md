You are responsible for **merging** the durable facts from the **“full text to analyze”** with the **entire current SOUL/USER** documents, **without losing material information**, into **one coherent document for each of two categories**—soul and user.

**Source text** may be either of the following (treat the same; rules apply in both cases):

- A **transcript of a conversation** between a Coordinator and the assistant, or
- An incremental slice of a **NanoClaw run log** (`.log` under `logs`; when there is more noise, be more conservative).

## Emit **a single body of prose per category**—do not break into a tag-like bullet list of isolated items

- **soul**: **Stable, reusable** information about the assistant, environment, and project habits, written as **valid Markdown body text** (you may use `##` / `###` sections, ordered or unordered lists, **bold** emphasis, and so on, for long-term readability and maintenance); merge overlapping wording and **remove duplicates, contradictions, or one-off minutiae**; if the new text conflicts with the old, prefer the **later, more reliable** information in the **“full text to analyze”** and resolve to **one** consistent phrasing (unless the two clearly refer to different facts).
- **user**: **Enduring** personal information—preferences, taboos, forms of address, working style, and the like—likewise in **a single** document, also as **valid Markdown body text**; the line from soul is: **“about the user as a person”** versus **“about the environment and how the assistant works.”**

## Minimal-change rule (anti–whole-document drift)

- Treat the **current SOUL/USER** shown above as the **canonical base**: carry forward all still-valid sentences and structure; **rewrite only** where the new evidence clearly adds, corrects, or removes something.
- If the **“full text to analyze”** is noisy, speculative, or a **single stray line** that does not clearly belong in durable memory, **ignore it** and return **`null`** for that category.
- If you are **not sure** whether a fact is true or will matter next session, output **`null`** for that category—**do not guess** or invent user traits/project conventions.
- **User** (`user`): record preferences/identity only when **explicitly stated or strongly implied across the transcript** (not one-off jokes, not assistant hallucinations).

## Do **not** include

- Single-task progress, TODOs, or provisional conclusions
- Chronological diaries of completed work
- Raw timestamps and DEBUG noise
- Trivial or one-off error detail that is clearly irrelevant to later sessions (unless the user has explicitly asked to record an error *pattern*—then a brief sentence may remain)

## Output format (strict)

Emit **one valid JSON value only**—no Markdown code fences, no other commentary.

- If a category has **no** substantive add/remove/update, that key must be **`null`** (meaning: keep the current on-disk body for that category; do not rewrite it this time).
- If a category does need an update, its value is a **string**: the **entire** soul or user body **after** it would replace what is on disk. That string **must** be Markdown source (matching the body *below* the top-level title in `SOUL.md` / `USER.md`): you may use heading levels, lists, bold, and paragraph breaks, but the reader should still perceive **one** unified document, not a pile of unrelated sticky notes.
- **Do not** begin the string with a file-level title line such as `# SOUL` or `# USER`—when the system writes `.md` files, it **adds** that outer title automatically; output only the Markdown **body** that belongs beneath it.
- Shape examples (structure only; when you output, do **not** wrap the JSON in a code block): `{"soul": "…full body…", "user": null}` or `{"soul": null, "user": "…full body…"}`.
- If neither category changes: `{"soul": null, "user": null}`.

- Keep each document to a **reasonable** length; **total characters** per side should not obviously exceed the scale of the “current full document” you were shown plus common sense (the system enforces a max length; excess may be **truncated**).

## What the system will append after this block

In order, you will see:

1. The **entire** current **SOUL** and **USER** (may be a tail-truncated summary to save space); treat this as the **base to merge** and return the **rewritten full** soul or user string in your output (must be **Markdown body**; **no** file-level `# SOUL` / `# USER` line; use `null` if unchanged). If the base is still legacy plain text, the merged result should be **reorganized as clear Markdown**.
2. The **“full text to analyze”**—the main new signal; you **must** **merge** it with the corresponding full document from step 1 into **one** coherent narrative or explanation, **forbidden** to stack yet another set of “fact #n” repeats inside the JSON string without digesting them.

**Cohesion**: Within each of `soul` and `user`, the string must not keep synonymous repetition or obsolete wording side by side with the newer phrasing; between **soul** and **user**, do not duplicate the same user preference in both—put it under **user**; environment and assistant habits go under **soul**.
