import type { APIRoute } from "astro";
import { getCollection } from "astro:content";

// Strips an entry's raw markdown body down to plain, searchable text.
function stripMarkdown(md: string): string {
  return md
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/<img[^>]*>/gi, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/^>\s*\[![\w-]+\]\s*/gm, "")
    .replace(/^>\s?/gm, "")
    .replace(/\${1,2}[^$]*\${1,2}/g, " ")
    .replace(/[#*_`~]/g, "")
    .replace(/^-{3,}\s*$/gm, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export const GET: APIRoute = async () => {
  const [posts, notes, books] = await Promise.all([
    getCollection("posts", ({ data }) => data.publish),
    getCollection("notes", ({ data }) => data.publish),
    getCollection("books", ({ data }) => data.publish),
  ]);

  const items = [
    ...posts.map((entry) => ({
      section: "posts" as const,
      title: entry.data.title,
      url: `/posts/${entry.id}`,
      tags: entry.data.tags,
      body: stripMarkdown(entry.body ?? ""),
    })),
    ...notes.map((entry) => ({
      section: "notes" as const,
      title: entry.data.title,
      url: `/notes/${entry.id}`,
      tags: entry.data.tags,
      body: stripMarkdown(entry.body ?? ""),
    })),
    ...books
      .filter((entry) => (entry.body ?? "").trim() !== "")
      .map((entry) => ({
        section: "books" as const,
        title: entry.data.title,
        url: `/books/${entry.id}`,
        meta: [entry.data.author, entry.data.originalTitle, entry.data.publisher]
          .filter(Boolean)
          .join(" · "),
      })),
  ];

  return new Response(JSON.stringify(items), {
    headers: { "Content-Type": "application/json" },
  });
};
