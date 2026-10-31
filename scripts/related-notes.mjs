// Related notes, scored across notes and posts together so a post can point
// at a note and vice versa. Shared tags count by rarity (a tag on 3 pages
// says more than one on 20), a wikilink in either direction counts most, and
// a shared affiliation/category only breaks ties. A page needs a shared tag
// or a link to be picked at all; category alone never qualifies.
//
// Kept apart from sync-vault.mjs so the scoring can be inspected on its own
// (each pick carries its score and reasons) without running a full sync.
export const NOTE_RELATED_LIMIT = 5;
const LINK_WEIGHT = 3;
const AFFILIATION_WEIGHT = 0.5;
const CATEGORY_WEIGHT = 0.3;

const asList = (v) => (Array.isArray(v) ? v : v ? [v] : []).map(String).filter(Boolean);
const overlap = (a, b) => [...a].filter((x) => b.has(x));

// items: [{ ref, title, file, fm, body }] — ref is what ends up in `related`.
// Returns Map<ref, [{ ref, score, sharedTags, linked, sameAffiliation, sameCategory }]>.
export function relateNotes(items) {
  const pool = items.map((it) => ({
    ref: it.ref,
    names: new Set([it.title, it.file.replace(/\.md$/, "")]),
    tags: new Set(asList(it.fm.tags)),
    affiliations: new Set(asList(it.fm.affiliation).map((a) => a.toLowerCase())),
    category: it.fm.category ?? null,
    links: new Set([...it.body.matchAll(/\[\[([^\]|#]+)/g)].map((m) => m[1].trim())),
    created: String(it.fm.created),
  }));
  const tagCount = new Map();
  for (const n of pool) for (const t of n.tags) tagCount.set(t, (tagCount.get(t) ?? 0) + 1);
  const idf = (t) => Math.log(pool.length / tagCount.get(t));
  const linksTo = (from, to) => [...to.names].some((name) => from.links.has(name));

  const result = new Map();
  for (const self of pool) {
    const picks = [];
    for (const other of pool) {
      if (other === self) continue;
      const sharedTags = overlap(self.tags, other.tags);
      const linked = linksTo(self, other) || linksTo(other, self);
      if (sharedTags.length === 0 && !linked) continue;
      const sameAffiliation = overlap(self.affiliations, other.affiliations).length > 0;
      const sameCategory = self.category != null && self.category === other.category;
      const score =
        sharedTags.reduce((sum, t) => sum + idf(t), 0) +
        (linked ? LINK_WEIGHT : 0) +
        (sameAffiliation ? AFFILIATION_WEIGHT : 0) +
        (sameCategory ? CATEGORY_WEIGHT : 0);
      picks.push({ ref: other.ref, created: other.created, score, sharedTags, linked, sameAffiliation, sameCategory });
    }
    picks.sort((x, y) => y.score - x.score || y.created.localeCompare(x.created));
    result.set(self.ref, picks.slice(0, NOTE_RELATED_LIMIT));
  }
  return result;
}
