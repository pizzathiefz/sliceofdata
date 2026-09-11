// @ts-check
import { defineConfig } from 'astro/config';
import cloudflare from '@astrojs/cloudflare';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import { remarkCallouts } from './remark-callouts.mjs';

// https://astro.build/config
export default defineConfig({
  adapter: cloudflare(),
  markdown: {
    remarkPlugins: [remarkMath, remarkCallouts],
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
