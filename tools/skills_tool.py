#!/usr/bin/env python3
"""Skills Tool — list and view skill documents (progressive disclosure). A skill is a directory
holding SKILL.md (YAML frontmatter + instructions) plus optional references/, templates/, assets/,
scripts/. `skills_list` returns name/description only; `skill_view` returns full content and
linked files. Sibling modules (skills_tool_setup / _plugin / _dedup) re-export here."""

import contextvars
import json
import logging
import os
import time
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error
from hermes_cli.config import cfg_get
from agent.skill_utils import (
    EXCLUDED_SKILL_DIRS as _EXCLUDED_SKILL_DIRS, is_skill_support_path as _is_skill_support_path)
from tools.skills_tool_setup import (
    SkillReadinessStatus, _build_setup_note, _capture_required_environment_variables,
    _get_required_environment_variables, _is_env_var_persisted, _is_remote_env_backend)
from tools.skills_tool_plugin import (
    MAX_DESCRIPTION_LENGTH, MAX_NAME_LENGTH, _INJECTION_PATTERNS, _fail, _json,
    _mark_background_review_read, _preprocess_skill, _read_skill_text, _safe_frontmatter,
    _serve_plugin_skill, _serve_skill_file, _truncate_description)
from tools.skills_tool_dedup import dedup_skill_view, reset_skill_view_dedup
from tools.skill_provenance import is_background_review

logger = logging.getLogger(__name__)

# Per-session discovery cache: {cache_key: (signature, timestamp, skills_list)}. Signature =
# every catalog directory's mtime and child-dir names, walked like the catalog itself (an install
# under an existing category such as ``mlops/training/`` moves only that dir's mtime) + the disabled
# set (config-only change, no mtime) + platform; the TTL bounds staleness from in-place SKILL.md
# edits, which no directory signature can see.
_SKILLS_CACHE: dict = {}
_SKILLS_CACHE_TTL_SECONDS = 30.0
# Deepest directory the signature descends to below a root (a skill sits at <cat>/<sub>/<skill>/).
_SIGNATURE_MAX_DEPTH = 6


def clear_skills_cache() -> None:
    """Drop the scanned catalog so the next listing re-scans (explicit reloads must not wait out the TTL)."""
    _SKILLS_CACHE.clear()


def _dir_tree_signature(root, sig, depth=_SIGNATURE_MAX_DEPTH) -> None:
    """Append ``(dir, mtime_ns, child dir names)`` for *root* and the dirs below it the catalog walks
    (``iter_skill_index_files``: no excluded dirs, no support dirs of a skill), depth-bounded. The child
    names, not only mtimes: an install and an uninstall inside one coarse mtime tick (1-2 s on some
    filesystems) leave every mtime equal, and the change must still show. ~1 ms for 170 dirs."""
    from agent.skill_utils import EXCLUDED_SKILL_DIRS, SKILL_SUPPORT_DIRS
    try:
        mtime = os.stat(root).st_mtime_ns
        with os.scandir(root) as it:
            entries = list(it)
    except OSError:
        return
    names, is_skill = [], False
    for entry in entries:
        with suppress(OSError):
            if entry.is_dir():
                names.append(entry.name)
            elif entry.name == "SKILL.md":
                is_skill = True
    names.sort()
    sig.append((str(root), mtime, tuple(names)))
    if depth <= 0:
        return
    for name in names:
        if name not in EXCLUDED_SKILL_DIRS and not (is_skill and name in SKILL_SUPPORT_DIRS):
            _dir_tree_signature(os.path.join(root, name), sig, depth - 1)


def _skills_scan_signature(dirs_to_scan, disabled) -> tuple:
    """Stat-based change signature over every directory the catalog walks; platform is read via
    ``agent.skill_utils.sys`` so test patches are honored."""
    from agent import skill_utils as _skill_utils
    platform = getattr(getattr(_skill_utils, "sys", None), "platform", "")
    sig: list = []
    for d in dirs_to_scan:
        _dir_tree_signature(str(d), sig)
    return (tuple(sig), frozenset(disabled), platform)


HERMES_HOME = get_hermes_home()  # all skills live in ~/.hermes/skills/ (seeded from bundled)
SKILLS_DIR = HERMES_HOME / "skills"
_SKILLS_DIR_AT_IMPORT = SKILLS_DIR


def _skills_dir() -> Path:
    """Active profile's skills dir at call time: the patched ``SKILLS_DIR`` when a patcher changed
    it, else live profile-scoped HERMES_HOME (long-lived runtimes may import before profile set)."""
    configured = Path(SKILLS_DIR)
    return configured if configured != _SKILLS_DIR_AT_IMPORT else get_hermes_home() / "skills"


_secret_capture_callback = None
_LOOKUP_HINT = "Use a skill name or relative path within the skills directory."


def skill_lookup_path_error(name: str) -> Optional[str]:
    """Error if lookup *name* could escape the search roots it is joined onto. Windows drive
    paths are rejected too: their ``:`` would be misread as a plugin namespace separator."""
    from tools.path_security import has_traversal_component
    if not isinstance(name, str):
        return "Skill name must be a string."
    win = PureWindowsPath(candidate := name.strip())
    if PurePosixPath(candidate).is_absolute() or win.is_absolute() or win.drive:
        return "Skill name must be a relative path within the skills directory."
    if has_traversal_component(candidate):
        return "Skill name cannot contain '..' path traversal components."
    return None


def load_env() -> dict[str, str]:
    """Snapshot of HERMES_HOME/.env for the post-skill secret-capture diff (same tokenizer that
    installs the profile scope, so a captured value never differs from the served one). A frozen-policy
    worker never opens the profile ``.env``: ``hermes_cli.config.load_env`` owns that short-circuit."""
    from agent.safe_worker_policy import worker_config_snapshot

    if worker_config_snapshot() is not None:
        return {}
    from agent.secret_scope import load_env_file

    return load_env_file(get_hermes_home() / ".env")


def set_secret_capture_callback(callback) -> None:
    global _secret_capture_callback
    _secret_capture_callback = callback


# A turn-scoped capture callback (the session owner binds one per local-viewer turn): it wins over the
# process-global slot and over the gateway-surface "unsupported" hint, because that turn's viewer can
# answer. A ContextVar, so tool worker threads started with ``propagate_context_to_thread`` inherit it.
_turn_secret_capture = contextvars.ContextVar("turn_secret_capture", default=None)


@contextmanager
def bind_turn_secret_capture(callback):
    token = _turn_secret_capture.set(callback)
    try:
        yield
    finally:
        _turn_secret_capture.reset(token)


def _skill_utils_delegate(attr: str):
    """Lazy call-time delegate to ``agent.skill_utils.<attr>`` (re-export; patches honored)."""
    def _delegate(*args):
        from agent import skill_utils
        return getattr(skill_utils, attr)(*args)
    _delegate.__name__ = _delegate.__qualname__ = attr
    return _delegate


skill_matches_platform = _skill_utils_delegate("skill_matches_platform")
# Offer-time relevance gate (kanban/docker/s6), NOT hard compatibility; explicit loads bypass it.
skill_matches_environment = _skill_utils_delegate("skill_matches_environment")
skill_matches_apps = _skill_utils_delegate("skill_matches_apps")
_parse_frontmatter = _skill_utils_delegate("parse_frontmatter")
_get_disabled_skill_names = _skill_utils_delegate("get_disabled_skill_names")


def check_skills_requirements() -> bool:
    return True  # always available: the directory is created on first use


def _get_category_from_path(skill_path: Path) -> Optional[str]:
    """``~/.hermes/skills/mlops/axolotl/SKILL.md`` -> ``"mlops"``; active profile dir first
    (respects test monkeypatching), then skills.external_dirs."""
    dirs_to_check = [_skills_dir()]
    with suppress(Exception):
        from agent.skill_utils import get_external_skills_dirs
        dirs_to_check.extend(get_external_skills_dirs())
    for skills_dir in dirs_to_check:
        with suppress(ValueError):
            if len(parts := skill_path.relative_to(skills_dir).parts) >= 3:
                return parts[0]
    return None


def _parse_tags(tags_value) -> list[str]:
    """Tags from frontmatter: a parsed list, "[a, b]", or "a, b"."""
    if not tags_value:
        return []
    if isinstance(tags_value, list):
        return [str(t).strip() for t in tags_value if t]
    tags_value = str(tags_value).strip()
    if tags_value.startswith("[") and tags_value.endswith("]"):
        tags_value = tags_value[1:-1]
    return [t.strip().strip("\"'") for t in tags_value.split(",") if t.strip()]


def _is_skill_disabled(*names: str, platform: str | None = None) -> bool:
    """Any of *names* disabled in config (one config load)? Platform precedence: explicit arg,
    ``HERMES_PLATFORM``, session ``HERMES_SESSION_PLATFORM``. A globally-disabled skill stays disabled on every platform
    (keep in sync with agent.skill_utils.get_disabled_skill_names)."""
    try:
        from hermes_cli.config import load_config
        skills_cfg = load_config().get("skills", {})
        resolved_platform = platform or os.getenv("HERMES_PLATFORM")
        if not resolved_platform:
            with suppress(Exception):
                from gateway.session_context import get_session_env
                resolved_platform = get_session_env("HERMES_SESSION_PLATFORM") or ""
        platform_disabled = None
        if resolved_platform:
            platform_disabled = cfg_get(skills_cfg, "platform_disabled", resolved_platform)
        disabled = skills_cfg.get("disabled", [])
        return any((platform_disabled is not None and n in platform_disabled) or n in disabled for n in names)
    except Exception:
        return False


def skill_search_dirs() -> tuple[list[tuple[int, Path]], Path]:
    """(``(tier, dir)`` roots in precedence order, active_skills_dir) — the shared
    ``agent.skill_utils.get_skill_search_roots`` order with the live profile dir (dropped if absent)."""
    from agent.skill_utils import TIER_LOCAL, get_skill_search_roots
    active_skills_dir = _skills_dir()
    roots = [(t, d) for t, d in get_skill_search_roots(active_skills_dir)
             if t != TIER_LOCAL or d.exists()]
    return roots, active_skills_dir


def _skill_catalog(*, skip_disabled: bool = False, include_hidden: bool = False) -> list[dict[str, Any]]:
    """Every scanned skill resolved by ``agent.skill_utils.resolve_skill_catalog`` (status /
    load_name / tier / path), visible ones only unless *include_hidden*; cached per session.
    Resolution runs over ALL files first — skill_view ignores platform/disabled gates when
    collecting candidates, so it asks for hidden rows too."""
    from agent.skill_utils import (
        TIER_PROJECT, is_disabled_entry, iter_project_skill_files, iter_skill_index_files, resolve_skill_catalog)
    cache_key = ("with_disabled" if skip_disabled else "filtered", include_hidden)
    disabled = set() if skip_disabled else _get_disabled_skill_names()
    roots, _ = skill_search_dirs()
    signature = _skills_scan_signature([d for _t, d in roots], disabled)
    now = time.monotonic()
    cached = _SKILLS_CACHE.get(cache_key)
    if cached is not None and cached[0] == signature and (now - cached[1]) < _SKILLS_CACHE_TTL_SECONDS:
        # Shallow copies: callers mutate the returned dicts (web_server annotates
        # s["enabled"]/s["usage"]); handing out cached objects would poison the cache.
        return [dict(s) for s in cached[2]]
    scanned = []
    for tier, scan_dir in roots:  # project dirs go through the quarantine chokepoint
        _iter = iter_project_skill_files if tier == TIER_PROJECT else lambda d: iter_skill_index_files(d, "SKILL.md")
        for skill_md in _iter(scan_dir):
            if any(part in _EXCLUDED_SKILL_DIRS for part in skill_md.parts):
                continue
            try:
                frontmatter, body = _parse_frontmatter(_read_skill_text(skill_md)[:4000])
                description = frontmatter.get("description", "")
                if not description:  # first non-heading body line (a null value stays null)
                    description = next((ln for ln in map(str.strip, body.strip().split("\n"))
                                        if ln and not ln.startswith("#")), description)
                scanned.append({
                    "name": frontmatter.get("name", skill_md.parent.name)[:MAX_NAME_LENGTH],
                    "description": _truncate_description(description),
                    "category": _get_category_from_path(skill_md), "tier": tier, "root": scan_dir,
                    "path": skill_md, "visible": bool(skill_matches_platform(frontmatter)
                                                      and skill_matches_environment(frontmatter)
                                                      and skill_matches_apps(frontmatter))})
            except (UnicodeDecodeError, PermissionError) as e:
                logger.debug("Failed to read skill file %s: %s", skill_md, e)
            except Exception as e:
                logger.debug("Skipping skill at %s: failed to parse: %s", skill_md, e, exc_info=True)
    skills = [s for s in resolve_skill_catalog(scanned)
              if (s.pop("visible") or include_hidden) and not is_disabled_entry(s, disabled)]
    # Keyed by the signature computed BEFORE the scan: a write racing the scan changes the
    # signature, so the next call re-scans instead of serving a torn result.
    _SKILLS_CACHE[cache_key] = (signature, now, skills)
    return [dict(s) for s in skills]


def find_all_skills(*, skip_disabled: bool = False) -> list[dict[str, Any]]:
    """Loadable skills (name, description, category): ``name`` is what skill_view() accepts —
    the declared name, or the exact relative path for a same-tier duplicate. Shadowed and
    unloadable copies are left out. ``skip_disabled=True`` ignores disabled state (config UI)."""
    return [{"name": s["load_name"], "description": s["description"], "category": s["category"]}
            for s in _skill_catalog(skip_disabled=skip_disabled) if s["load_name"]]


def sort_skills(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep every skill listing path ordered the same way."""
    return sorted(skills, key=lambda s: (s.get("category") or "", s["name"]))


def skills_list(category: str | None = None, task_id: str | None = None) -> str:
    """Tier 1 listing: name + description (+ category) only; ``task_id`` is handler parity."""
    try:
        _skills_dir().mkdir(parents=True, exist_ok=True)
        all_skills = find_all_skills()
        try:
            from hermes_cli.plugins import discover_plugins, get_plugin_manager
            discover_plugins()
            for plugin_skill in get_plugin_manager().list_plugin_skill_metadata():
                frontmatter = plugin_skill.pop("frontmatter", {})
                if not skill_matches_platform(frontmatter) or _is_skill_disabled(plugin_skill["name"]):
                    continue
                all_skills.append(plugin_skill)
        except Exception:
            logger.debug("Plugin skill listing failed", exc_info=True)
        if not all_skills:
            return _json({"success": True, "skills": [], "categories": [],
                          "message": "No skills found in skills/ directory."})
        if category:
            all_skills = [s for s in all_skills if s.get("category") == category]
        all_skills = sort_skills(all_skills)
        categories = sorted({s.get("category") for s in all_skills if s.get("category")})
        return _json({
            "success": True, "skills": all_skills, "categories": categories,
            "count": len(all_skills),
            "hint": "Use skill_view(name) to see full content, tags, and linked files"})
    except Exception as e:
        return tool_error(str(e), success=False)


def _legacy_portable_skill_name(pm, namespace, bare):
    """Portable plugin skills were named ``agent-plugin-<slug>-<sha8>:<skill>`` before they took the
    plugin's own name; prompts and cron jobs saved since then may still carry that form. Resolved
    through what that exact plugin registered, so a skill it lost to a name clash is not found."""
    if not namespace.startswith("agent-plugin-"):
        return None
    keys = {key for key, loaded in pm._plugins.items()
            if loaded.manifest.portable and loaded.manifest.data_namespace == namespace}
    return next((qualified for qualified, entry in pm._plugin_skills.items()
                 if entry.get("plugin_key") in keys and entry.get("bare_name") == bare), None)


def _resolve_plugin_skill(name, file_path, task_id, preprocess):
    """``plugin:skill`` dispatch: ``(result_json, None)`` when answered, else ``(None,
    local_category_name)`` to fall through to the flat-tree scan — categorized local skills also use
    ``category:skill`` in config/gateway prompts, so the on-disk ``category/skill`` form returns."""
    from agent.skill_utils import is_valid_namespace, parse_qualified_name
    from hermes_cli.plugins import discover_plugins, get_plugin_manager
    namespace, bare = parse_qualified_name(name)
    if namespace == "plugin" and ":" in bare:
        # The skills index lists plugin skills under a "plugin" heading, and models write that
        # heading as a prefix ("plugin:agent-plugin-blender-5a80e4ba:blender"). A bare skill name
        # never holds a colon, so the remainder is the real qualified name.
        name = bare
        namespace, bare = parse_qualified_name(name)
    if not is_valid_namespace(namespace):
        return _fail(f"Invalid namespace '{namespace}' in '{name}'. Namespaces must match [a-zA-Z0-9_-]+."), None
    discover_plugins()  # idempotent
    pm = get_plugin_manager()
    if legacy := _legacy_portable_skill_name(pm, namespace, bare):
        name = legacy
        namespace = legacy.partition(":")[0]
    active_memory_provider = None
    try:
        from plugins.memory import _get_active_memory_provider, _prune_inactive_memory_provider_skills
        active_memory_provider = _get_active_memory_provider()
        _prune_inactive_memory_provider_skills(active_memory_provider)
    except Exception as exc:
        logger.debug("Failed pruning inactive memory-provider skills: %s", exc)
    plugin_skill_md = pm.find_plugin_skill(name)
    # Memory providers load through plugins.memory, not the general PluginManager: load the
    # namespaced provider once so its collector can forward its skills into the registry.
    if plugin_skill_md is None and namespace == active_memory_provider:
        try:
            from plugins.memory import load_memory_provider
            load_memory_provider(namespace)
            plugin_skill_md = pm.find_plugin_skill(name)
        except Exception as exc:
            logger.debug("Failed lazy memory-provider skill load for %s: %s", namespace, exc)
    if plugin_skill_md is not None and not plugin_skill_md.exists():
        pm.remove_plugin_skill(name)  # stale registry entry — file deleted out of band
        return _fail(
            f"Skill '{name}' file no longer exists at {plugin_skill_md}. The registry entry "
            f"has been cleaned up — try again after the plugin is reloaded."), None
    if plugin_skill_md is not None:
        return _serve_plugin_skill(
            plugin_skill_md, namespace, bare, file_path=file_path, preprocess=preprocess, session_id=task_id), None
    if available := pm.list_plugin_skills(namespace):  # plugin exists but this specific skill is missing
        return _fail(
            f"Skill '{bare}' not found in plugin '{namespace}'.",
            available_skills=[f"{namespace}:{s}" for s in available],
            hint=f"The '{namespace}' plugin provides {len(available)} skill(s)."), None
    return None, (f"{namespace}/{bare}" if bare else None)  # plugin not found → local scan


def _under_any(path: Path, dirs) -> bool:
    """True when ``path`` (resolved where possible) lives under one of ``dirs``."""
    resolved = path
    with suppress(Exception):
        resolved = path.resolve()
    return any(resolved.is_relative_to(d) for d in dirs)


def _is_package_owned_markdown(path: Path, search_root: Path) -> bool:
    """True when a legacy Markdown candidate belongs to an ancestor directory skill."""
    try:
        relative = path.relative_to(search_root)
    except ValueError:
        return False
    return any(
        (search_root.joinpath(*relative.parts[:depth]) / "SKILL.md").is_file()
        for depth in range(1, len(relative.parts))
    )


def _collect_skill_candidates(name, local_category_name, all_dirs):
    """ALL (skill_dir, skill_md) candidates across every dir and lookup strategy (direct path,
    recursive by dir / frontmatter name, legacy flat <name>.md), deduped by resolved path.
    Every copy is returned so the caller resolves them via ``agent.skill_utils.pick_skill_candidate``
    (cross-tier precedence; a same-tier tie of different skills is refused, never guessed)."""
    from agent.skill_utils import iter_skill_index_files
    candidates: list[tuple[Optional[Path], Path]] = []
    seen_md: set = set()

    def _record(sd: Optional[Path], smd: Path) -> None:
        key = smd
        with suppress(Exception):
            key = smd.resolve()
        if key not in seen_md:
            seen_md.add(key)
            candidates.append((sd, smd))

    def _record_direct(direct_path: Path, search_root: Path) -> None:  # "mlops/axolotl" / "axolotl" or its flat .md sibling
        flat = direct_path.with_suffix(".md")
        if not _is_skill_support_path(direct_path) and direct_path.is_dir() and (direct_path / "SKILL.md").exists():
            _record(direct_path, direct_path / "SKILL.md")
        elif (flat.exists() and not _is_skill_support_path(flat)
              and not _is_package_owned_markdown(flat, search_root)):
            _record(None, flat)

    for search_dir in all_dirs:
        for direct in filter(None, (name, local_category_name)):  # "p:x" with no plugin p → "p/x"
            _record_direct(search_dir / direct, search_dir)
        # Recursive by directory name plus frontmatter `name:` — skills_list()
        # exposes the frontmatter name, so skill_view(name) must accept it too.
        for found_skill_md in iter_skill_index_files(search_dir, "SKILL.md"):
            if (found_skill_md.parent.name == name
                    or _safe_frontmatter(found_skill_md).get("name") == name):
                _record(found_skill_md.parent, found_skill_md)
        # Legacy flat <name>.md anywhere under the dir. Markdown owned by an ancestor
        # directory skill loads through file_path and must not shadow a real skill.
        for found_md in search_dir.rglob(f"{name}.md"):
            if (found_md.name != "SKILL.md" and not _is_skill_support_path(found_md)
                    and not _is_package_owned_markdown(found_md, search_dir)):
                _record(None, found_md)
    return candidates


# (support dir, globs, recursive, files only) — order is the linked_files key order.
_LINKED_FILE_SPECS = (
    ("references", ["*.md"], False, False),
    ("templates", ["*.md", "*.py", "*.yaml", "*.yml", "*.json", "*.tex", "*.sh"], True, False),
    ("assets", ["*"], True, True),
    ("scripts", ["*.py", "*.sh", "*.bash", "*.js", "*.ts", "*.rb"], False, False))


def _skill_linked_files(skill_dir: Optional[Path]) -> dict:
    """references/templates/assets/scripts of a directory skill (empty groups dropped)."""
    files: dict = {}
    for sub, globs, recursive, files_only in _LINKED_FILE_SPECS if skill_dir else ():
        base = skill_dir / sub
        found = [
            f.relative_to(skill_dir).as_posix() for g in globs if base.exists()
            for f in (base.rglob(g) if recursive else base.glob(g))
            if not files_only or f.is_file()]
        if found:
            files[sub] = found
    return files


def _skill_readiness(frontmatter: dict[str, Any], skill_name: str) -> tuple[dict, dict]:
    """Resolve required env vars / credential files (prompting for secrets where the surface
    allows) and register what's available for sandboxes. Returns ``(fields, extras)``: fields go
    before the readiness extras in the skill_view result — key order is tool output."""
    required_env_vars = _get_required_environment_variables(frontmatter)
    from tools.terminal_scope import terminal_env
    backend = str(terminal_env("TERMINAL_ENV", "local")).strip().lower() or "local"
    env_snapshot = load_env()
    missing_required_env_vars = [
        e for e in required_env_vars
        if not e.get("optional") and not _is_env_var_persisted(e["name"], env_snapshot)]
    capture_result = _capture_required_environment_variables(skill_name, missing_required_env_vars)
    if missing_required_env_vars:  # re-read: a successful capture persisted into .env
        env_snapshot = load_env()
    still_missing = set(capture_result["missing_names"])
    remaining = [
        e["name"] for e in required_env_vars if not e.get("optional")
        and (e["name"] in still_missing or not _is_env_var_persisted(e["name"], env_snapshot))]
    setup_needed = bool(remaining)
    # Only vars actually set pass through to sandboxed execution (execute_code, terminal).
    if available_env_names := [e["name"] for e in required_env_vars if e["name"] not in remaining]:
        try:
            from tools.env_passthrough import register_env_passthrough
            register_env_passthrough(available_env_names)
        except Exception:
            logger.debug("Could not register env passthrough for skill %s", skill_name, exc_info=True)
    # Credential files for remote sandboxes: existing host files are registered,
    # missing ones flag setup_needed.
    required_cred_files_raw = frontmatter.get("required_credential_files", [])
    missing_cred_files: list = []
    if isinstance(required_cred_files_raw, list) and required_cred_files_raw:
        try:
            from tools.credential_files import register_credential_files
            missing_cred_files = register_credential_files(required_cred_files_raw)
            setup_needed = setup_needed or bool(missing_cred_files)
        except Exception:
            logger.debug("Could not register credential files for skill %s", skill_name, exc_info=True)
    status = SkillReadinessStatus.SETUP_NEEDED if setup_needed else SkillReadinessStatus.AVAILABLE
    fields = {
        "required_environment_variables": required_env_vars, "required_commands": [],
        "missing_required_environment_variables": remaining,
        "missing_credential_files": missing_cred_files, "missing_required_commands": [],
        "setup_needed": setup_needed, "setup_skipped": capture_result["setup_skipped"],
        "readiness_status": status.value}
    extras: dict = {}
    if setup_help := next((e["help"] for e in required_env_vars if e.get("help")), None):
        extras["setup_help"] = setup_help
    if capture_result["gateway_setup_hint"]:
        extras["gateway_setup_hint"] = capture_result["gateway_setup_hint"]
    missing_items = [f"env ${n}" for n in remaining] + [f"file {p}" for p in missing_cred_files]
    if setup_needed and (setup_note := _build_setup_note(status, missing_items, setup_help)):
        if _is_remote_env_backend(backend):
            setup_note = f"{setup_note} {backend.upper()}-backed skills need these requirements available inside the remote environment as well."
        extras["setup_note"] = setup_note
    return fields, extras


def _owning_search_dir(skill_md: Path, all_dirs) -> Optional[Path]:
    """Most specific search dir containing *skill_md*, compared lexically: a symlinked entry
    belongs to the root that exposes it, not to the root its target lives in."""
    owners = [Path(d) for d in all_dirs if skill_md.is_relative_to(d)]
    return max(owners, key=lambda d: len(d.parts), default=None)


def locate_skill(name: str, local_category_name: Optional[str], roots):
    """Unique on-disk skill for *name* over ``(tier, dir)`` *roots*: cross-tier precedence
    (project > local > create_dir > external, shadowed copies logged), same-tier collision refusal,
    same-root identical-copy ranking, quarantine gate, not-found listing. ``(error_json, skill_dir,
    skill_md)``; skill_md set iff no error."""
    from agent.skill_utils import AMBIGUOUS_SKILL_PREFIX, TIER_PROJECT, pick_skill_candidate, skill_candidate_rank
    all_dirs = [d for _t, d in roots]
    if not all_dirs:
        return _fail(
            "Skills directory does not exist yet. It will be created on first install."), None, None
    candidates = _collect_skill_candidates(name, local_category_name, all_dirs)
    tier_of = {d: t for t, d in roots}
    if len(candidates) > 1:
        owned = [(c, _owning_search_dir(c[1], all_dirs)) for c in candidates]
        won, contenders = pick_skill_candidate([
            (tier_of[root], str(root), skill_candidate_rank(c[1], root), c[1]) for c, root in owned])
        if won is not None:
            if dropped := [str(smd) for i, (_sd, smd) in enumerate(candidates) if i != won]:
                logger.info("Skill '%s' resolved to %s by precedence (project > local > create_dir > "
                            "external_dirs, then identical same-root copies); not loaded: %s",
                            name, candidates[won][1], "; ".join(dropped))
            candidates = [candidates[won]]
        else:
            candidates = [candidates[i] for i in contenders]
    if len(candidates) > 1:
        paths = [str(smd) for _, smd in candidates]
        # A root-level copy's path IS the bare name, so it can't disambiguate itself.
        load_names = sorted({_owned_relative(sd, smd, all_dirs) for sd, smd in candidates} - {name})
        logger.warning("Skill name collision for '%s': %d candidates — %s", name, len(candidates), "; ".join(paths))
        return _fail(
            f"{AMBIGUOUS_SKILL_PREFIX}'{name}': "
            + (f"use one of {', '.join(load_names)}. " if load_names else "")
            + f"{len(candidates)} different skills share this name in the same skills directory tier; "
            "refusing to guess.",
            matches=paths, load_names=load_names,
            hint="Pass the exact relative path instead of the bare name, or rename one of the "
            "colliding skills so each name is unique."), None, None
    skill_dir, skill_md = candidates[0] if candidates else (None, None)
    # Quarantine gate: a project-tier skill with a dangerous scan verdict must not
    # load even by explicit name (same chokepoint the index and skills_list use).
    if skill_md is not None and tier_of.get(_owning_search_dir(skill_md, all_dirs)) == TIER_PROJECT:
        from agent.skill_utils import is_quarantined_project_skill
        if is_quarantined_project_skill(skill_md):
            return _fail(
                f"Project skill '{name}' is quarantined: the security scan flagged its content as "
                "dangerous. It will not load until the repo's skill content changes and passes a re-scan.",
                hint="Inspect the skill in the repo checkout, or untrust the repo with "
                "`hermes skills untrust`."), None, None
    if not skill_md or not skill_md.exists():
        available = [s["name"] for s in sort_skills(find_all_skills())[:20]]
        return _fail(f"Skill '{name}' not found.", available_skills=available,
                     hint="Use skills_list to see all available skills"), None, None
    return None, skill_dir, skill_md


def _owned_relative(skill_dir: Optional[Path], skill_md: Path, all_dirs) -> str:
    """Exact load path of a candidate: its skill dir (or flat ``.md`` stem) relative to its root."""
    root = _owning_search_dir(skill_md, all_dirs)
    target = skill_dir if skill_dir is not None else skill_md.with_suffix("")
    return target.relative_to(root).as_posix() if root is not None else str(target)


def _log_security_warnings(name: str, skill_md: Path, content: str, all_dirs, active_skills_dir):
    """Warn (never block) when loaded from outside the trusted dirs (project + local + external)
    and/or when common prompt-injection patterns appear. The check is on the RESOLVED path:
    every candidate is built as ``<search_dir>/...`` so a lexical test can never fire, and a
    SKILL.md symlinked to a file outside every root is exactly what this guards against."""
    trusted_dirs = [active_skills_dir.resolve()]
    with suppress(Exception):
        trusted_dirs.extend(d.resolve() for d in all_dirs)
    warnings = []
    if not _under_any(skill_md, trusted_dirs):
        warnings.append(f"skill file is outside the trusted skills directory (~/.hermes/skills/): {skill_md}")
    if any(p in content.lower() for p in _INJECTION_PATTERNS):
        warnings.append("skill content contains patterns that may indicate prompt injection")
    if warnings:
        logger.warning("Skill security warning for '%s': %s", name, "; ".join(warnings))


def skill_view(
    name: str, file_path: str | None = None, task_id: str | None = None, preprocess: bool = True) -> str:
    """View a skill (SKILL.md) or a file within its directory, as JSON. ``name`` is a skill name
    or path ("axolotl", "03-fine-tuning/axolotl"); "<plugin>:<skill>" resolves plugin-provided
    skills. ``preprocess`` applies the configured SKILL.md template / inline shell rendering;
    slash/preload callers render the message themselves."""
    # The top-level agent never receives a skill body: every later turn in that
    # session would pay for it as input tokens. Workers (depth >= 1) still get the
    # document, because their context is discarded after the call.
    from tools.agent_context import agent_depth

    if file_path is None and agent_depth() == 0:
        return json.dumps(
            {
                "success": False,
                "error": (
                    "skill_view is disabled for the top-level agent: a skill document must "
                    "not enter this context. Call skill_run(name=..., request=...) instead -- "
                    "it executes the skill in a subagent and returns only that subagent's report."
                ),
                "hint": "Use skill_find(query) to discover a skill, then skill_run(name, request).",
            },
            ensure_ascii=False,
        )
    try:
        # Validate before the ':' dispatch so a Windows drive path (C:\skills\foo) can't be
        # reinterpreted as a plugin namespace.
        if lookup_error := skill_lookup_path_error(name):
            return _fail(lookup_error, hint=_LOOKUP_HINT)
        local_category_name: str | None = None
        if ":" in name:  # plugin registry; bare names use the flat-tree scan below
            served, local_category_name = _resolve_plugin_skill(name, file_path, task_id, preprocess)
            if served is not None:
                return served
        # The fall-through form (namespace/bare) joins onto each search dir too; re-validate it
        # since `bare` is not namespace-checked.
        if local_category_name and (lookup_error := skill_lookup_path_error(local_category_name)):
            return _fail(lookup_error, hint=_LOOKUP_HINT)
        roots, active_skills_dir = skill_search_dirs()
        all_dirs = [d for _t, d in roots]
        error, skill_dir, skill_md = locate_skill(name, local_category_name, roots)
        if error is not None:
            return error
        try:  # read once — reused for platform check and main content
            content = _read_skill_text(skill_md)
        except Exception as e:
            return _fail(f"Failed to read skill '{name}': {e}")
        _log_security_warnings(name, skill_md, content, all_dirs, active_skills_dir)
        frontmatter = _safe_frontmatter(content=content)
        if not skill_matches_platform(frontmatter):
            return _fail(f"Skill '{name}' is not supported on this platform.", readiness_status=SkillReadinessStatus.UNSUPPORTED.value)
        resolved_name = frontmatter.get("name", skill_md.parent.name)
        # Disabled by declared name, or by the exact path that IS this copy's load name (a same-tier
        # duplicate's row) — whatever alias reached it; an unrelated copy at that path elsewhere is not.
        rel = _owned_relative(skill_dir, skill_md, all_dirs)
        if _is_skill_disabled(resolved_name, rel) and (_is_skill_disabled(resolved_name) or any(
                e["path"] == skill_md and e["load_name"] == rel for e in _skill_catalog(skip_disabled=True, include_hidden=True))):
            return _fail(f"Skill '{resolved_name}' is disabled. Enable it with `hermes skills` or inspect the files directly on disk.")
        if file_path and skill_dir:
            return _serve_skill_file(
                skill_dir, file_path, name, list_available=True, mark_read=True,
                hint="Use a relative path within the skill directory")
        # tags/related_skills: metadata.hermes.* (agentskills.io) first, then top-level.
        metadata = frontmatter.get("metadata")
        hermes_meta = (metadata.get("hermes", {}) or {}) if isinstance(metadata, dict) else {}
        tags, related_skills = (
            _parse_tags(hermes_meta.get(k) or frontmatter.get(k, "")) for k in ("tags", "related_skills"))
        linked_files = _skill_linked_files(skill_dir)
        try:
            rel_path = str(skill_md.relative_to(active_skills_dir))
        except ValueError:  # external skill — relative to its own parent dir
            rel_path = str(skill_md.relative_to(skill_md.parent.parent)) if skill_md.parent.parent else skill_md.name
        skill_name = frontmatter.get("name", skill_md.stem if not skill_dir else skill_dir.name)
        readiness, readiness_extras = _skill_readiness(frontmatter, skill_name)
        rendered_content = content if not preprocess else _preprocess_skill(
            content, skill_dir, task_id, "Could not preprocess skill content for %s", skill_name)
        # ── pm tool deps (`deps: [ffmpeg]` frontmatter) ──────────────
        # Loading the skill IS the activation moment: ensure each declared
        # pm package now so the skill's commands work when the model runs
        # them. Failure never blocks the skill content — the note carries
        # the remedy.
        deps_note = None
        declared_deps = frontmatter.get("deps") or []
        if isinstance(declared_deps, str):
            declared_deps = [declared_deps]
        if isinstance(declared_deps, list) and declared_deps:
            failed_deps = []
            for dep in [str(d).strip() for d in declared_deps if str(d).strip()]:
                try:
                    import pm

                    pm.ensure(dep)
                except Exception as exc:
                    failed_deps.append(f"{dep}: {exc}")
            if failed_deps:
                deps_note = (
                    "Tool dependencies could not be installed — "
                    + "; ".join(failed_deps)
                    + ". Run `hermes pm install "
                    + " ".join(str(d) for d in declared_deps)
                    + "` and reload."
                )

        result = {
            "success": True, "name": skill_name, "description": frontmatter.get("description", ""),
            "tags": tags, "related_skills": related_skills, "content": rendered_content,
            "path": rel_path, "skill_dir": str(skill_dir) if skill_dir else None,
            "linked_files": linked_files if linked_files else None,
            "usage_hint": "To view linked files, call skill_view(name, file_path) where file_path is e.g. 'references/api.md' or 'assets/config.yaml'" if linked_files else None,
            **readiness,
            **readiness_extras}
        if deps_note:
            result["deps_note"] = deps_note
        _mark_background_review_read(skill_md)
        if frontmatter.get("compatibility"):  # agentskills.io optional fields
            result["compatibility"] = frontmatter["compatibility"]
        if isinstance(metadata, dict):
            result["metadata"] = metadata
        return _json(result)
    except Exception as e:
        return tool_error(str(e), success=False)


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": "List available skills (name + description). Use skill_view(name) to load full content.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter to narrow results",
            }
        },
        "required": [],
    },
}

SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": "Load a skill's SKILL.md, or one of its linked references, templates or scripts. The first call returns SKILL.md plus a 'linked_files' dict; call again with file_path to read one.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The skill name (use skills_list to see available skills). For plugin-provided skills, use the qualified form '<plugin>:<skill>' (e.g. 'superpowers:writing-plans').",
            },
            "file_path": {
                "type": "string",
                "description": "OPTIONAL: Path to a linked file within the skill (e.g., 'references/api.md', 'templates/config.yaml', 'scripts/validate.py'). Omit to get the main SKILL.md content.",
            },
        },
        "required": ["name"],
    },
}

registry.register(
    name="skills_list", toolset="skills", schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list(category=args.get("category"), task_id=kw.get("task_id")),
    check_fn=check_skills_requirements, emoji="📚")


def _skill_view_with_bump(args, **kw):
    """Invoke skill_view, then bump view_count/use on success (best-effort). Repeat-view dedup
    mirrors read_file's unchanged-stub: a result byte-identical to one already served for the
    same skill file in this session returns a short stub (cache cleared on context compression)."""
    name = args.get("name", "")
    task_id = kw.get("task_id")
    # The background-review fork shares the parent's task_id (prefix-cache parity). A stub there
    # (a) skips the read-mark its read-before-write guard requires and (b) lets it patch from a
    # possibly-pruned transcript copy (#95976). No dedup in the fork; None also keeps its views
    # out of the parent's bucket.
    dedup_task_id = None if is_background_review() else task_id
    result = skill_view(name, file_path=args.get("file_path"), task_id=task_id)
    with suppress(Exception):
        parsed = json.loads(result)
        if isinstance(parsed, dict) and parsed.get("success"):
            if (stub := dedup_skill_view(dedup_task_id, name, args.get("file_path"), parsed, result)) is not None:
                return stub
            if resolved := parsed.get("name") or name:  # qualified forms return the canonical name
                from tools.skill_usage import bump_use, bump_view
                bump_view(str(resolved))
                # Viewing is actively loading the skill to act on it — that counts as use
                # (the curator's stale timer keys off last_used_at).
                bump_use(str(resolved), task_id=kw.get("task_id"), session_id=kw.get("session_id"))
    return result


registry.register(
    name="skill_view", toolset="skills", schema=SKILL_VIEW_SCHEMA, handler=_skill_view_with_bump,
    check_fn=check_skills_requirements, emoji="📚")


# ---------------------------------------------------------------------------
# skill_run / skill_find: execute a skill in a subagent, keep the body out of the
# parent. skill_view above is the only way to obtain the text, and it refuses at
# depth 0, so the document can only ever reach a worker's context.
# ---------------------------------------------------------------------------

WORKER_PREAMBLE = (
    "You are a skill worker: a disposable agent executing one skill for a caller that "
    "cannot see it. The complete skill document follows. Treat it as the authority on "
    "method; where it is wrong or impossible in this environment, do the reasonable thing "
    "and say so in notes. Actually do the work -- every claim in your report needs real "
    "command output or an artifact behind it. No user is present: never ask questions; if "
    "something essential is missing, return status=blocked and say what you need. Files "
    "inside the skill directory can be opened with "
    "skill_view(name='<skill>', file_path='references/whatever.md'). Return ONLY a JSON "
    "object with keys: status (ok|partial|blocked), answer (the text the caller should "
    "relay), result, evidence (array), artifacts (array), notes.\n\n"
    "===== SKILL DOCUMENT BEGINS =====\n"
)

_STOPWORDS = frozenset(
    "a an the and or of for to in on at with by from is are be do does did please can could "
    "would should i you we my me it this that these those how what when where why which use "
    "using make get need want help new".split()
)


def _skill_roots() -> List[Path]:
    """Skill directories in precedence order.

    ``skill_search_dirs()`` returns ``([(tier, Path), ...], active_dir)``, so unwrap
    the tier when present.
    """
    try:
        roots, _active = skill_search_dirs()
    except Exception:
        roots = []
    out: List[Path] = []
    for item in roots or []:
        d = Path(item[1] if isinstance(item, tuple) else item)
        if d.is_dir() and d not in out:
            out.append(d)
    return out


def _skill_md_for(name: str) -> Optional[Path]:
    """Locate a skill's SKILL.md by directory name, then by frontmatter name."""
    for root in _skill_roots():
        direct = root / name
        if (direct / "SKILL.md").is_file():
            return direct / "SKILL.md"
        for md in root.rglob("SKILL.md"):
            if md.parent.name == name:
                return md
    for root in _skill_roots():
        for md in root.rglob("SKILL.md"):
            try:
                from agent.skill_utils import parse_frontmatter

                fm, _body = parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if str(fm.get("name") or "") == name:
                return md
    return None


def _tokens(text: str) -> set:
    import re

    return {w for w in re.findall(r"[a-z0-9]{3,}", text.lower()) if w not in _STOPWORDS}


def skill_run(name: str, request: str, task_id: str = None, **kw) -> str:
    """Execute one skill inside a subagent; return only that subagent's report."""
    from tools.agent_context import current_agent

    parent = current_agent()
    if parent is None:
        return tool_error("skill_run must be called by an agent (no agent context bound).",
                          success=False)
    if not name or not request:
        return tool_error("skill_run requires both 'name' and 'request'.", success=False)

    skill_md = _skill_md_for(str(name))
    if skill_md is None:
        return tool_error(f"No skill named {name!r}.", success=False)
    try:
        body = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return tool_error(f"Could not read {skill_md}: {exc}", success=False)
    if len(body) > 200_000:
        body = body[:200_000] + "\n\n[... truncated at 200k chars ...]"

    from tools.delegate_tool import delegate_task

    return delegate_task(
        goal=f"Execute the '{name}' skill for this request: {request}",
        context=WORKER_PREAMBLE + body + "\n===== SKILL DOCUMENT ENDS =====\n\n"
                                        f"Request: {request}",
        parent_agent=parent,
        background=False,
    )


def skill_find(query: str, limit: int = 5, task_id: str = None, **kw) -> str:
    """Deterministic keyword ranking over the catalog. Names and purposes only."""
    import re

    if not query:
        return tool_error("skill_find requires 'query'.", success=False)
    try:
        limit = max(1, min(int(limit or 5), 25))
    except (TypeError, ValueError):
        limit = 5
    from agent.skill_utils import parse_frontmatter

    q = _tokens(str(query))
    scored: List[Tuple[float, str, str]] = []
    for root in _skill_roots():
        for md in root.rglob("SKILL.md"):
            try:
                fm, _body = parse_frontmatter(md.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            sname = str(fm.get("name") or md.parent.name)
            purpose = re.sub(r"\s+", " ", str(fm.get("description") or "")).strip()
            score = 3.0 * len(q & _tokens(sname.replace("-", " ").replace("_", " "))) + \
                1.0 * len(q & _tokens(purpose))
            if sname.lower() in str(query).lower():
                score += 5.0
            if score > 0:
                scored.append((round(score, 2), sname, purpose))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return json.dumps(
        [{"skill": n, "score": s, "purpose": (p[:90] + "...") if len(p) > 90 else p}
         for s, n, p in scored[:limit]],
        ensure_ascii=False,
    )


SKILL_RUN_SCHEMA = {
    "name": "skill_run",
    "description": (
        "Execute one of the installed skills. The skill document is loaded inside a "
        "subagent, never into this conversation; only that subagent's JSON report comes "
        "back. Use this instead of reading a skill, and instead of applying a skill's "
        "procedure yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Skill name, e.g. 'arxiv'."},
            "request": {"type": "string",
                        "description": "The concrete task, phrased so a worker can act on it alone."},
        },
        "required": ["name", "request"],
    },
}

SKILL_FIND_SCHEMA = {
    "name": "skill_find",
    "description": (
        "Find which installed skills can do a job. Deterministic keyword ranking over "
        "skill names and descriptions; returns names, scores and short purposes only -- "
        "never a skill body. Follow with skill_run(name, request)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What needs doing, in a few words."},
            "limit": {"type": "integer", "description": "Max results (default 5)."},
        },
        "required": ["query"],
    },
}

registry.register(
    name="skill_run", toolset="skills", schema=SKILL_RUN_SCHEMA,
    handler=lambda args, **kw: skill_run(args.get("name", ""), args.get("request", ""),
                                        task_id=kw.get("task_id"), **kw),
    check_fn=check_skills_requirements, emoji="📚")
registry.register(
    name="skill_find", toolset="skills", schema=SKILL_FIND_SCHEMA,
    handler=lambda args, **kw: skill_find(args.get("query", ""), limit=args.get("limit", 5),
                                          task_id=kw.get("task_id")),
    check_fn=check_skills_requirements, emoji="📚")
