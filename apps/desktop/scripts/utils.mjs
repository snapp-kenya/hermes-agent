
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Returns true if the passed file is being invoked from node, not imported.
// `node -e` / stdin runs have no argv[1]: nothing is main.
//
// Compare real paths, not resolved ones: macOS TMPDIR is /var/folders/... which
// is a symlink to /private/var/folders/..., so a script reached through the
// symlinked form has an import.meta.url the plain resolve() can never match.
// That mismatch made the entry script exit 0 without running: a silent no-op
// that produced unstamped artifacts instead of an error.
/** @param {string} importMetaUrl @returns {boolean} */
export function isMain(importMetaUrl) {
    const entry = process.argv[1];
    if (entry === undefined) {
        return false;
    }
    const identity = (candidate) => {
        try {
            return fs.realpathSync(candidate);
        } catch {
            return null;
        }
    };
    const self = identity(fileURLToPath(importMetaUrl));
    return self !== null && self === identity(path.resolve(entry));
}
