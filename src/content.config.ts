import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    tags: z.array(z.string()).default([]),
    created: z.coerce.date(),
    publish: z.boolean().default(true),
  }),
});

const books = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/books" }),
  schema: z.object({
    title: z.string(),
    author: z.string().optional(),
    publisher: z.string().optional(),
    yearOfPublication: z.coerce.number().optional(),
    originalTitle: z.string().optional(),
    created: z.coerce.date(),
    rating: z.coerce.number().min(0).max(5).optional(),
    publish: z.boolean().default(true),
  }),
});

const notes = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/notes" }),
  schema: z.object({
    title: z.string(),
    tags: z.array(z.string()).default([]),
    created: z.coerce.date(),
    publish: z.boolean().default(true),
  }),
});

const films = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/films" }),
  schema: z.object({
    title: z.string(),
    year: z.coerce.number().optional(),
    created: z.coerce.date(),
    rating: z.coerce.number().min(0).max(5).optional(),
    comment: z.string().optional(),
    publish: z.boolean().default(true),
  }),
});

export const collections = { posts, books, films, notes };
