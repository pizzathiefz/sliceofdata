// Keeps src/content in step with the vault during `astro dev`: syncs once on
// startup (so `astro dev` works without the npm predev hook) and again
// whenever a vault note or asset changes. Builds still sync via prebuild.
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { VAULT_ROOT } from "./vault-path.mjs";

const sync = (logger) => {
  try {
    execFileSync(process.execPath, ["scripts/sync-vault.mjs"], { stdio: "pipe" });
    logger.info("vault synced");
  } catch (err) {
    logger.error(`vault sync failed: ${err.stderr || err.message}`);
  }
};

export default function vaultSync() {
  return {
    name: "vault-sync",
    hooks: {
      "astro:config:setup": ({ command, logger }) => {
        if (command === "dev") sync(logger);
      },
      "astro:server:setup": ({ logger }) => {
        let timer;
        const watchers = ["content", "assets"].map((dir) =>
          fs.watch(path.join(VAULT_ROOT, dir), { recursive: true }, (_event, file) => {
            // Skip Obsidian/iCloud temp files; they churn without real edits.
            if (file && /(^|\/)\.|\.icloud$/.test(file)) return;
            clearTimeout(timer);
            timer = setTimeout(() => sync(logger), 500);
          })
        );
        process.once("exit", () => watchers.forEach((w) => w.close()));
      },
    },
  };
}
