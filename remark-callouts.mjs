import { visit } from "unist-util-visit";

// remarkCallouts runs before remarkBreaks, so continuation lines are still
// literal "\n" inside the text node here — the title capture must stop at
// the first newline, or a titleless callout swallows its whole body as the
// title (since \s* matches \n and .* + $ then consume the rest of the node).
const CALLOUT_RE = /^\[!(\w+)\]([+-])?[ \t]*([^\n]*)/;

// Overrides the auto-generated (capitalized-type) label for types whose
// display label isn't just Title Case of the type name.
const DEFAULT_LABELS = { tldr: "TL;DR" };

export function remarkCallouts() {
  return (tree) => {
    visit(tree, "blockquote", (node) => {
      const firstChild = node.children[0];
      if (!firstChild || firstChild.type !== "paragraph") return;
      const firstText = firstChild.children[0];
      if (!firstText || firstText.type !== "text") return;

      const match = firstText.value.match(CALLOUT_RE);
      if (!match) return;

      const [, type, , title] = match;
      const rest = firstText.value.slice(match[0].length).replace(/^\n/, "");

      if (rest) {
        firstText.value = rest;
      } else {
        firstChild.children.shift();
        if (firstChild.children[0]?.type === "break") firstChild.children.shift();
      }

      const kind = type.toLowerCase();
      const label =
        title || DEFAULT_LABELS[kind] || type.charAt(0).toUpperCase() + type.slice(1).toLowerCase();

      node.data = node.data || {};
      node.data.hName = "div";
      node.data.hProperties = { className: ["callout", `callout-${kind}`] };

      node.children.unshift({
        type: "paragraph",
        data: { hName: "div", hProperties: { className: ["callout-title"] } },
        children: [{ type: "text", value: label }],
      });
    });
  };
}
