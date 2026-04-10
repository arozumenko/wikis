#!/usr/bin/env node
/**
 * wikis skill installer.
 *
 * Three ways to consume the wikis skill:
 *
 *   1. Claude Code plugin marketplace (preferred inside Claude Code):
 *        /plugin marketplace add arozumenko/wikis
 *        /plugin install wikis@wikis
 *
 *   2. This CLI (works for Claude Code, Cursor, Windsurf, GitHub Copilot —
 *      copies skills directly into the IDE dirs of the current project):
 *        npx github:arozumenko/wikis init
 *        npx github:arozumenko/wikis init --all
 *        npx github:arozumenko/wikis init --skills wikis
 *        npx github:arozumenko/wikis init --target claude
 *        npx github:arozumenko/wikis init --update    # overwrite existing
 *
 *   3. agentskills.io / Vercel / any third-party tool:
 *        npx skills add https://github.com/arozumenko/wikis
 *      The spec-compliant SKILL.md is at skills/wikis/SKILL.md.
 */

import { cpSync, existsSync, mkdirSync, readdirSync, statSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { createInterface } from "readline";

const __dirname = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = join(__dirname, "..");
const CWD = process.cwd();

const TARGETS = [
  { id: "claude",  dir: ".claude",   label: "Claude Code"     },
  { id: "cursor",  dir: ".cursor",   label: "Cursor"          },
  { id: "windsurf",dir: ".windsurf", label: "Windsurf"        },
  { id: "copilot", dir: ".github",   label: "GitHub Copilot"  },
];

// ---------------------------------------------------------------------------
// Catalog discovery — reads skills/ at runtime; no hardcoded lists.
// ---------------------------------------------------------------------------

function listDirs(parent) {
  const root = join(PKG_ROOT, parent);
  if (!existsSync(root)) return [];
  return readdirSync(root)
    .filter((n) => !n.startsWith(".") && n !== "README.md")
    .filter((n) => {
      try { return statSync(join(root, n)).isDirectory(); } catch { return false; }
    })
    .sort();
}

function loadCatalog() {
  return { skills: listDirs("skills") };
}

// ---------------------------------------------------------------------------
// CLI parsing
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  const out = { all: false, update: false, yes: false, skills: null, targets: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if      (a === "--all")     out.all = true;
    else if (a === "--yes")     out.yes = true;
    else if (a === "--update")  out.update = true;
    else if (a === "--skills")  out.skills  = splitList(argv[++i]);
    else if (a === "--target")  out.targets = splitList(argv[++i]);
    else if (a === "--help" || a === "-h") { printHelp(); process.exit(0); }
  }
  return out;
}

function splitList(value) {
  if (!value) return [];
  return value.split(",").map((s) => s.trim()).filter(Boolean);
}

function printHelp() {
  console.log(`
  wikis skill installer

  Usage:
    npx github:arozumenko/wikis init [options]

  Options:
    --all                      Install all skills (no prompts)
    --skills  <a,b,c|all>      Install only these skills (or all)
    --target <claude,cursor,…> Limit IDE targets (default: all detected)
    --update                   Overwrite existing installs
    --yes                      Skip the interactive IDE prompt
    -h, --help                 Show this help

  Examples:
    npx github:arozumenko/wikis init
    npx github:arozumenko/wikis init --all
    npx github:arozumenko/wikis init --skills wikis --target claude
    npx github:arozumenko/wikis init --all --update
`);
}

// ---------------------------------------------------------------------------
// Install logic
// ---------------------------------------------------------------------------

function resolveSelection(requested, available, kind) {
  if (requested === null) return null;
  if (requested.length === 0) return [];
  if (requested.length === 1 && requested[0] === "all") return available;
  const unknown = requested.filter((r) => !available.includes(r));
  if (unknown.length) {
    console.error(`  ! Unknown ${kind}: ${unknown.join(", ")}`);
    console.error(`    Available: ${available.join(", ") || "(none)"}`);
    process.exit(1);
  }
  return requested;
}

function copyItem(name, targetDir, update) {
  const src  = join(PKG_ROOT, "skills", name);
  if (!existsSync(src)) return { status: "missing" };
  const dest = join(CWD, targetDir, "skills", name);
  if (existsSync(dest) && !update) return { status: "exists", dest };
  mkdirSync(dirname(dest), { recursive: true });
  cpSync(src, dest, { recursive: true, force: update });
  return { status: "installed", dest };
}

function ask(rl, q) {
  return new Promise((resolve) => rl.question(q, resolve));
}

async function interactivePick(catalog, args) {
  const detected = TARGETS.filter((t) => existsSync(join(CWD, t.dir)));

  let targets;
  if (args.targets) {
    targets = TARGETS.filter((t) => args.targets.includes(t.id));
    if (targets.length === 0) {
      console.error(`  ! No valid --target values: ${args.targets.join(", ")}`);
      process.exit(1);
    }
  } else if (args.all || args.yes) {
    targets = detected.length > 0 ? detected : [TARGETS[0]];
  } else {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    try {
      if (detected.length === 0) {
        console.log("  No IDE directories detected. Installing to .claude/");
        targets = [TARGETS[0]];
      } else {
        console.log("  Detected IDE directories:");
        detected.forEach((t, i) => console.log(`    ${i + 1}. ${t.label} (${t.dir}/)`));
        console.log("    a. All of the above\n");
        const choice = ((await ask(rl, "  Install to which? [a]: ")).trim().toLowerCase()) || "a";
        targets = choice === "a" ? detected : [detected[parseInt(choice) - 1] || detected[0]];
      }
    } finally {
      rl.close();
    }
  }

  let skillsSelection = resolveSelection(args.skills, catalog.skills, "skill");
  if (args.all) {
    if (skillsSelection === null) skillsSelection = catalog.skills;
  } else if (skillsSelection === null) {
    console.log("\n  No --skills specified. Installing all skills.");
    skillsSelection = catalog.skills;
  }

  return { targets, skillsSelection };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const args    = parseArgs(process.argv.slice(2));
  const catalog = loadCatalog();

  console.log("\n  wikis — AI-powered code documentation skill\n");
  console.log(`  Catalog: ${catalog.skills.length} skill(s)`);

  if (catalog.skills.length === 0) {
    console.log("\n  ! No skills found. Nothing to install.\n");
    return;
  }

  const { targets, skillsSelection } = await interactivePick(catalog, args);

  console.log("");
  let installed = 0, skipped = 0;

  for (const t of targets) {
    console.log(`  → ${t.label} (${t.dir}/)`);
    for (const name of skillsSelection) {
      const r = copyItem(name, t.dir, args.update);
      if      (r.status === "installed") { console.log(`      ✓ skill  ${name}`);                         installed++; }
      else if (r.status === "exists")    { console.log(`      — skill  ${name} (exists; use --update)`);  skipped++;   }
      else                               { console.log(`      ! skill  ${name} (missing in repo)`);                    }
    }
  }

  console.log(
    `\n  Done: ${installed} installed, ${skipped} skipped.` +
    (installed > 0 ? "\n  Set WIKIS_URL before using the skill.\n  Launch your IDE in this project to activate it." : "") +
    "\n"
  );
}

main().catch((err) => {
  console.error("Install failed:", err.message);
  process.exit(1);
});
