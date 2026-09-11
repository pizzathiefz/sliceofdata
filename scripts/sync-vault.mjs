// Syncs published content from the Obsidian vault (a separate git repo) into
// src/content/*, so Astro's glob() loader can pick it up like any other
// local content collection. Runs automatically before `astro dev`/`build`
// (see package.json) — there's nothing to do by hand.
//
// VAULT_PATH points at a local checkout of the vault repo. Locally this
// defaults to the iCloud-synced Obsidian vault itself; in CI it should be set
// to wherever the vault repo was freshly cloned before this script runs.
import fs from "node:fs";
import path from "node:path";

const VAULT_ROOT =
  process.env.VAULT_PATH ??
  "/Users/yshin/Library/Mobile Documents/iCloud~md~obsidian/Documents/notes";
const VAULT = path.join(VAULT_ROOT, "content");

function parseFrontmatter(raw) {
  const match = raw.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return null;
  const lines = match[1].split("\n");
  const data = {};
  let currentKey = null;
  for (const line of lines) {
    const listMatch = line.match(/^\s*-\s*(.+)$/);
    if (listMatch && currentKey) {
      if (!Array.isArray(data[currentKey])) data[currentKey] = [];
      data[currentKey].push(listMatch[1].trim());
      continue;
    }
    const kvMatch = line.match(/^([^:]+):\s*(.*)$/);
    if (kvMatch) {
      const key = kvMatch[1].trim();
      const value = kvMatch[2].trim();
      currentKey = key;
      if (value === "") {
        data[key] = undefined; // will become array via following list items
      } else {
        let v = value;
        if (key !== "comment") {
          let quoted;
          while ((quoted = v.match(/^"(.*)"$|^'(.*)'$/))) {
            v = quoted[1] ?? quoted[2];
          }
        }
        data[key] = v;
      }
    }
  }
  const body = raw.slice(match[0].length).replace(/^\n+/, "");
  return { fm: data, body };
}

function readCollection(folder, { requireRating = false, minYear = 2000 } = {}) {
  const dir = path.join(VAULT, folder);
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md"));

  const entries = [];
  for (const file of files) {
    const raw = fs.readFileSync(path.join(dir, file), "utf-8");
    const parsed = parseFrontmatter(raw);
    if (!parsed) continue;
    const { fm, body } = parsed;
    if (!fm.created) continue;
    if (new Date(fm.created).getFullYear() < minYear) continue;
    if (new Date(fm.created) > new Date()) continue;
    if (requireRating && !(Number(fm.rating) > 0)) continue;
    entries.push({ file, fm, body });
  }
  return entries.sort((a, b) => new Date(b.fm.created) - new Date(a.fm.created));
}

function readPublishedCollection(folder) {
  const dir = path.join(VAULT, folder);
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md"));

  const entries = [];
  for (const file of files) {
    const raw = fs.readFileSync(path.join(dir, file), "utf-8");
    const parsed = parseFrontmatter(raw);
    if (!parsed) continue;
    const { fm, body } = parsed;
    if (!fm.created) continue;
    if (fm.publish !== "true") continue;
    entries.push({ file, fm, body });
  }
  return entries.sort((a, b) => new Date(b.fm.created) - new Date(a.fm.created));
}

// Wiki notes (vault/content/wiki) are short glossary/definition entries with
// no page of their own on the site — they only ever appear inlined via a
// ![[Title]] embed in another note, so index them by title for that lookup.
function readWikiNotes() {
  const dir = path.join(VAULT, "wiki");
  const files = fs.readdirSync(dir).filter((f) => f.endsWith(".md") && f !== "_index.md");

  const map = new Map();
  for (const file of files) {
    const raw = fs.readFileSync(path.join(dir, file), "utf-8");
    const parsed = parseFrontmatter(raw);
    if (!parsed) continue;
    const { fm, body } = parsed;
    const title = fm.title ?? path.basename(file, ".md");
    map.set(title, { body, file, title });
  }
  return map;
}

function slugify(str) {
  const slug = str
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
  return slug || "untitled";
}

function makeSlugger() {
  const used = new Set();
  return (title) => {
    const base = slugify(title);
    let slug = base;
    let n = 2;
    while (used.has(slug)) {
      slug = `${base}-${n++}`;
    }
    used.add(slug);
    return slug;
  };
}

// Promotes single-line $$formula$$ into a genuine multi-line $$ block, which
// is the only form remark-math renders in KaTeX display (centered) mode.
function normalizeMathBlocks(body) {
  return body.replace(/^([ \t]*)\$\$(.+)\$\$[ \t]*$/gm, "$1$$$$\n$1$2\n$1$$$$");
}

const IMAGE_EXT_RE = /\.(png|jpe?g|gif|svg|webp)$/i;

// macOS's filesystem is case-insensitive, so a vault folder/file whose case
// drifted from the note's title/filename (e.g. Obsidian creating "deep &
// cross network" for a note titled "Deep & Cross Network") still resolves
// locally — but silently fails on the case-sensitive Linux runner that CI
// builds run on, leaving the raw ![[...]] embed unresolved in production.
// These two helpers do the same directory/file lookups case-insensitively so
// local and CI behavior match.
const ASSETS_ROOT = path.join(VAULT_ROOT, "assets");
let assetDirsByLowerName;
function resolveAssetDir(name) {
  if (!assetDirsByLowerName) {
    assetDirsByLowerName = new Map();
    for (const entry of fs.readdirSync(ASSETS_ROOT, { withFileTypes: true })) {
      if (entry.isDirectory()) assetDirsByLowerName.set(entry.name.toLowerCase(), entry.name);
    }
  }
  const real = assetDirsByLowerName.get(name.toLowerCase());
  return real ? path.join(ASSETS_ROOT, real) : null;
}
function findFileCaseInsensitive(dir, filename) {
  const direct = path.join(dir, filename);
  if (fs.existsSync(direct)) return direct;
  const lower = filename.toLowerCase();
  const match = fs.readdirSync(dir).find((f) => f.toLowerCase() === lower);
  return match ? path.join(dir, match) : null;
}

// Resolves ![[filename|width]] image embeds: copies the referenced asset from
// the vault's assets/ folder into public/<kind>/<slug>/, and rewrites the
// embed into a raw <img> tag (Astro's markdown pipeline passes raw HTML
// through, so the width hint survives).
//
// The attachment folder is named after the *title* for most notes, but posts
// use a short English slug as their filename while keeping the attachment
// folder under the original (often Korean) title — and some notes have had
// their title edited without renaming the file, leaving the folder matching
// only the filename. `names` is the ordered list of candidate folder names
// to try — normally just [title, filename], but a note being inlined via
// ![[..]] also carries the host note's own names, since Obsidian resolves
// ![[img]] by a vault-wide filename search and an image referenced from an
// embedded note is often actually sitting in the folder of whichever note
// first pasted it.
//
// Only embeds with an image extension are touched here — a bare ![[Title]]
// embed (no extension) is a note transclusion, handled by resolveEmbeds.
function resolveImages(body, kind, slug, names) {
  const candidateDirs = [...new Set(names)].map(resolveAssetDir).filter(Boolean);
  const publicDir = path.join("public", kind, slug);
  const withImages = body.replace(/!\[\[([^|\]]+)(?:\|(\d+))?\]\]/g, (whole, filename, width) => {
    if (!IMAGE_EXT_RE.test(filename.trim())) return whole;
    // the embed text is sometimes a bare filename and sometimes a full
    // "assets/<title>/..." path (Obsidian writes either depending on vault
    // settings) — assets always live flat under their folder, so only the
    // basename matters regardless of which form was written.
    const baseFilename = path.basename(filename);
    const srcPath = candidateDirs
      .map((dir) => findFileCaseInsensitive(dir, baseFilename))
      .find(Boolean);
    if (!srcPath) return whole;
    fs.mkdirSync(publicDir, { recursive: true });
    fs.copyFileSync(srcPath, path.join(publicDir, baseFilename));
    const widthAttr = width ? ` width="${width}"` : "";
    return `<img src="/${kind}/${slug}/${baseFilename}"${widthAttr} alt="" />`;
  });
  // an <img> HTML block absorbs any immediately-following line as raw HTML
  // (no blank line = same block in CommonMark) — force a blank line after so
  // a caption paragraph right below it still parses as markdown.
  return withImages.replace(/(<img[^>]*\/>)\n(?!\n)/g, "$1\n\n");
}

// Extracts one section (a heading and everything under it, up to the next
// heading of the same or shallower level) out of a body of markdown, by
// exact heading-text match. Returns null if no heading matches.
function extractSection(body, heading) {
  const lines = body.split("\n");
  const headingRe = /^(#{1,6})\s+(.*)$/;
  let startIdx = -1;
  let level = 0;
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(headingRe);
    if (m && m[2].trim() === heading) {
      startIdx = i;
      level = m[1].length;
      break;
    }
  }
  if (startIdx === -1) return null;
  let endIdx = lines.length;
  for (let i = startIdx + 1; i < lines.length; i++) {
    const m = lines[i].match(headingRe);
    if (m && m[1].length <= level) {
      endIdx = i;
      break;
    }
  }
  return lines.slice(startIdx, endIdx).join("\n").trim();
}

// Resolves ![[Title]] / ![[Title#Heading]] note-transclusion embeds by
// inlining the target's own content in place — the closest equivalent to
// Obsidian's live transclusion, since the site has no per-block embed
// rendering of its own. Works for a title from any collection (the vault's
// wiki/ glossary notes as well as published books/notes/posts): a #Heading
// suffix extracts just that section, otherwise the whole body is inlined.
// The target's own images are resolved against the host page's kind/slug,
// falling back to the host note's own asset folder for any image the
// target's folder doesn't have (Obsidian resolves ![[img]] by a vault-wide
// filename search, so an image referenced from an embedded note is often
// actually sitting in whichever note first pasted it). Anything that
// doesn't match a known title falls back to plain display text.
function resolveEmbeds(body, kind, slug, hostNames, depth = 0) {
  if (depth > 5) return body;
  return body.replace(/!\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (whole, target, alias) => {
    if (IMAGE_EXT_RE.test(target.trim())) return whole;
    const hashIdx = target.indexOf("#");
    const key = (hashIdx === -1 ? target : target.slice(0, hashIdx)).trim();
    const heading = hashIdx === -1 ? null : target.slice(hashIdx + 1).trim();
    const display = (alias ?? key).trim();

    const entry = embedMap.get(key);
    if (!entry) return display;

    const names = [entry.title, path.basename(entry.file, ".md"), ...hostNames];
    let content = resolveImages(entry.body, kind, slug, names);
    if (heading) {
      const section = extractSection(content, heading);
      if (section === null) return display;
      content = section;
    }
    return resolveEmbeds(content, kind, slug, hostNames, depth + 1);
  });
}

// Resolves [[Note Title]] / [[Note Title|display]] wikilinks into real links
// to the matching book/note/post page. A wikilink with no matching page (e.g.
// it points at a private or unpublished vault note) is rendered as plain
// text instead of the raw brackets.
function resolveWikilinks(body) {
  return body.replace(/(?<!!)\[\[([^\]|]+)(?:\|([^\]]+))?\]\]/g, (whole, target, alias) => {
    const key = target.trim();
    const display = (alias ?? target).trim();

    // "YYYY 📚" / "YYYY 🎬" are the vault's yearly index notes — point them at
    // the matching year tab on the books/films listing instead of leaving
    // them unresolved (those index notes themselves aren't published).
    const yearIndex = key.match(/^(\d{4})\s*(📚|🎬)$/);
    if (yearIndex) {
      const kind = yearIndex[2] === "📚" ? "books" : "films";
      return `[${display}](/${kind}#${yearIndex[1]})`;
    }

    const hit = linkMap.get(key);
    return hit ? `[${display}](/${hit.kind}/${hit.slug})` : display;
  });
}

function cleanBody(body, kind, slug, title, file) {
  const names = [title, path.basename(file, ".md")];
  return resolveWikilinks(
    resolveEmbeds(normalizeMathBlocks(resolveImages(body, kind, slug, names)), kind, slug, names)
  );
}

const wikiNotes = readWikiNotes();
const films = readCollection("film", { requireRating: true });
const books = readCollection("book");
const notes = readPublishedCollection("note");
const posts = readPublishedCollection("post");

console.log(
  `films: ${films.length}, books: ${books.length}, notes: ${notes.length}, posts: ${posts.length}`
);

// Precompute output slugs up front (in the same order/sluggers used below)
// so the link map below and the write loops agree on the same slug per
// entry without calling any slugger twice.
const filmSlug = makeSlugger();
for (const e of films) {
  e.title = e.fm.title ?? path.basename(e.file, ".md");
  e.slug = filmSlug(e.title);
}
const bookSlug = makeSlugger();
for (const e of books) {
  e.title = e.fm.title ?? path.basename(e.file, ".md");
  e.slug = bookSlug(e.title);
}
const noteSlug = makeSlugger();
for (const e of notes) {
  e.title = e.fm.title ?? path.basename(e.file, ".md");
  e.slug = noteSlug(e.title);
}
const postSlug = makeSlugger();
for (const e of posts) {
  e.title = e.fm.title ?? path.basename(e.file, ".md");
  e.slug = postSlug(path.basename(e.file, ".md"));
}

// Films have no individual page (just the /films listing), so they're left
// out of the map — a wikilink to a film falls back to plain text.
const linkMap = new Map();
// embedMap backs resolveEmbeds' ![[Title]] lookups — unlike linkMap it also
// carries the target's raw body (so it can be inlined) and includes the
// vault's wiki/ glossary notes, which have no page of their own to link to.
const embedMap = new Map();
for (const [title, w] of wikiNotes) {
  embedMap.set(title, { body: w.body, file: w.file, title: w.title });
}
for (const [entries, kind] of [
  [books, "books"],
  [notes, "notes"],
  [posts, "posts"],
]) {
  for (const e of entries) {
    const filenameKey = path.basename(e.file, ".md");
    linkMap.set(filenameKey, { kind, slug: e.slug });
    if (e.title !== filenameKey) linkMap.set(e.title, { kind, slug: e.slug });
    const embedEntry = { body: e.body, file: e.file, title: e.title };
    embedMap.set(filenameKey, embedEntry);
    if (e.title !== filenameKey) embedMap.set(e.title, embedEntry);
  }
}

const outDirs = {
  films: "src/content/films",
  books: "src/content/books",
  notes: "src/content/notes",
  posts: "src/content/posts",
};

for (const [kind, dir] of Object.entries(outDirs)) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
  fs.rmSync(path.join("public", kind), { recursive: true, force: true });
}

films.forEach(({ fm, title, slug }) => {
  const fmOut = [
    "---",
    `title: ${JSON.stringify(title)}`,
    fm.year ? `year: ${fm.year}` : null,
    `created: ${fm.created}`,
    fm.rating != null ? `rating: ${fm.rating}` : null,
    fm.comment ? `comment: ${JSON.stringify(fm.comment)}` : null,
    `publish: true`,
    "---",
    "",
  ]
    .filter(Boolean)
    .join("\n");
  fs.writeFileSync(path.join(outDirs.films, `${slug}.md`), fmOut);
});

books.forEach(({ fm, body, title, slug, file }) => {
  const author = Array.isArray(fm.author) ? fm.author.join(", ") : fm.author;
  const fmOut = [
    "---",
    `title: ${JSON.stringify(title)}`,
    author ? `author: ${JSON.stringify(author)}` : null,
    fm.publisher ? `publisher: ${JSON.stringify(fm.publisher)}` : null,
    fm["year of publication"] ? `yearOfPublication: ${fm["year of publication"]}` : null,
    fm["original title"] && fm["original title"] !== fm.title
      ? `originalTitle: ${JSON.stringify(fm["original title"])}`
      : null,
    `created: ${fm.created}`,
    fm.rating != null ? `rating: ${fm.rating}` : null,
    `publish: true`,
    "---",
    "",
    cleanBody(body, "books", slug, title, file),
  ]
    .filter((line) => line !== null)
    .join("\n");
  fs.writeFileSync(path.join(outDirs.books, `${slug}.md`), fmOut);
});

notes.forEach(({ fm, body, title, slug, file }) => {
  const tags = Array.isArray(fm.tags) ? fm.tags : fm.tags ? [fm.tags] : [];
  const fmOut = [
    "---",
    `title: ${JSON.stringify(title)}`,
    tags.length ? `tags: [${tags.map((t) => JSON.stringify(t)).join(", ")}]` : null,
    `created: ${fm.created}`,
    `publish: true`,
    "---",
    "",
    cleanBody(body, "notes", slug, title, file),
  ]
    .filter((line) => line !== null)
    .join("\n");
  fs.writeFileSync(path.join(outDirs.notes, `${slug}.md`), fmOut);
});

posts.forEach(({ fm, body, title, slug, file }) => {
  const tags = Array.isArray(fm.tags) ? fm.tags : fm.tags ? [fm.tags] : [];
  const fmOut = [
    "---",
    `title: ${JSON.stringify(title)}`,
    tags.length ? `tags: [${tags.map((t) => JSON.stringify(t)).join(", ")}]` : null,
    `created: ${fm.created}`,
    `publish: true`,
    "---",
    "",
    cleanBody(body, "posts", slug, title, file),
  ]
    .filter((line) => line !== null)
    .join("\n");
  fs.writeFileSync(path.join(outDirs.posts, `${slug}.md`), fmOut);
});

console.log("done");
