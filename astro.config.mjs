// @ts-check
import { defineConfig } from 'astro/config';
import cloudflare from '@astrojs/cloudflare';
import remarkMath from 'remark-math';
import remarkBreaks from 'remark-breaks';
import rehypeKatex from 'rehype-katex';
import { remarkCallouts } from './remark-callouts.mjs';

// https://astro.build/config
export default defineConfig({
  adapter: cloudflare(),
  markdown: {
    // remarkBreaks: Obsidian renders a single newline as a line break;
    // standard markdown collapses it into a space.
    remarkPlugins: [remarkMath, remarkCallouts, remarkBreaks],
    rehypePlugins: [rehypeKatex],
    shikiConfig: {
      themes: {
        light: 'catppuccin-latte',
        dark: 'catppuccin-mocha',
      },
      defaultColor: false,
    },
  },
});
