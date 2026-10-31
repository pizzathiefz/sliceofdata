import { getCollection } from "astro:content";

// Heading of the related box on note and post pages (books use "more from the shelf").
export const RELATED_LABEL = "because you read this";

// Resolves a page's `related` refs ("notes/<slug>" / "posts/<slug>") into
// shelf items.
export async function relatedShelfItems(related: string[]) {
  const pages = {
    notes: await getCollection("notes", ({ data }) => data.publish),
    posts: await getCollection("posts"),
  };
  return related.flatMap((ref) => {
    const [kind, id] = ref.split("/") as [keyof typeof pages, string];
    const page = pages[kind]?.find((p) => p.id === id);
    return page ? [{ href: `/${ref}`, title: page.data.title }] : [];
  });
}
