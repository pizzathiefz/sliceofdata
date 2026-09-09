# sliceofdata

Personal site built with [Astro](https://astro.build), covering four sections: **posts** (long-form writing), **notes** (short paper/article notes), **books**, and **films**.

## Content

All content lives in a separate Obsidian vault repo (`notes`), not in this repo. On every `dev`/`build`, [`scripts/sync-vault.mjs`](scripts/sync-vault.mjs) reads the vault's `content/` and `assets/` folders, converts Obsidian-specific syntax (image embeds, single-line `$$math$$`) into standard Markdown/HTML, and writes the result into `src/content/*` for Astro's content collections to pick up. `src/content/*` and the synced `public/{films,books,notes,posts}` folders are gitignored — the vault repo is the source of truth.

By default the sync reads from the local iCloud-synced vault path. Set `VAULT_PATH` (see [`.env.example`](.env.example)) to point it at a different checkout, e.g. a freshly cloned copy of the vault repo in CI.

## Commands

| Command           | Action                                                         |
| :----------------- | :-------------------------------------------------------------- |
| `npm install`       | Install dependencies                                             |
| `npm run dev`       | Sync content from the vault, then start the dev server           |
| `npm run build`     | Sync content from the vault, then build to `./dist/`             |
| `npm run sync`      | Just run the vault sync, without starting/building anything      |
| `npm run preview`   | Preview a production build locally                               |

## Stack

Astro (Content Layer collections), KaTeX for math, Shiki for code blocks (dual light/dark themes), custom remark plugins for Obsidian-style callouts (`> [!note]`).
