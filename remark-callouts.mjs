import { visit } from "unist-util-visit";

const CALLOUT_RE = /^\[!(\w+)\]([+-])?\s*(.*)$/;

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
      const label = title || type.charAt(0).toUpperCase() + type.slice(1).toLowerCase();

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
