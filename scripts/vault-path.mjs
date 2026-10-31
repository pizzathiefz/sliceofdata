// VAULT_PATH points at a local checkout of the vault repo. Locally this
// defaults to the iCloud-synced Obsidian vault itself; in CI it should be set
// to wherever the vault repo was freshly cloned before syncing.
export const VAULT_ROOT =
  process.env.VAULT_PATH ??
  "/Users/yshin/Library/Mobile Documents/iCloud~md~obsidian/Documents/notes";
