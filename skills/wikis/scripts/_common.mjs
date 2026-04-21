/**
 * Shared helpers for Wikis skill scripts.
 *
 * Reads WIKIS_URL and WIKIS_TOKEN from the environment and exports
 * ready-to-use values for all scripts in this directory.
 */

export const wikisUrl = (() => {
  const url = process.env.WIKIS_URL;
  if (!url) {
    console.error("Error: WIKIS_URL environment variable is not set.");
    console.error("  export WIKIS_URL=http://localhost:3000");
    process.exit(1);
  }
  return url.replace(/\/$/, "");
})();

export const headers = {
  Accept: "application/json",
  ...(process.env.WIKIS_TOKEN
    ? { Authorization: `Bearer ${process.env.WIKIS_TOKEN}` }
    : {}),
};

/**
 * Print an error and exit with code 1.
 * @param {string} message - Human-readable error message.
 * @param {string} [detail] - Optional raw response body for context.
 */
export function die(message, detail) {
  console.error(`Error: ${message}`);
  if (detail) {
    try {
      console.error(JSON.stringify(JSON.parse(detail), null, 2));
    } catch {
      console.error(detail);
    }
  }
  process.exit(1);
}
