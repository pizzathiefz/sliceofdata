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

// Resolves ![[filename|width]] image embeds: copies the referenced asset from
// the vault's assets/ folder into public/<kind>/<slug>/, and rewrites the
// embed into a raw <img> tag (Astro's markdown pipeline passes raw HTML
// through, so the width hint survives).
//
// The attachment folder is named after the *title* for most notes, but posts
// use a short English slug as their filename while keeping the attachment
// folder under the original (often Korean) title — and some notes have had
// their title edited without renaming the file, leaving the folder matching
// only the filename. Try both and use whichever one actually has the file.
function resolveImages(body, kind, slug, title, file) {
  const candidateDirs = [...new Set([title, path.basename(file, ".md")])].map((name) =>
    path.join(VAULT_ROOT, "assets", name)
  );
  const publicDir = path.join("public", kind, slug);
  const withImages = body.replace(/!\[\[([^|\]]+)(?:\|(\d+))?\]\]/g, (whole, filename, width) => {
    // the embed text is sometimes a bare filename and sometimes a full
    // "assets/<title>/..." path (Obsidian writes either depending on vault
    // settings) — assets always live flat under their folder, so only the
    // basename matters regardless of which form was written.
    const baseFilename = path.basename(filename);
    const srcPath = candidateDirs
      .map((dir) => path.join(dir, baseFilename))
      .find((p) => fs.existsSync(p));
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
  return resolveWikilinks(normalizeMathBlocks(resolveImages(body, kind, slug, title, file)));
}

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
for (const [entries, kind] of [
  [books, "books"],
  [notes, "notes"],
  [posts, "posts"],
]) {
  for (const e of entries) {
    const filenameKey = path.basename(e.file, ".md");
    linkMap.set(filenameKey, { kind, slug: e.slug });
    if (e.title !== filenameKey) linkMap.set(e.title, { kind, slug: e.slug });
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
