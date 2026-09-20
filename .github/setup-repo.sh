#!/usr/bin/env bash
# =============================================================================
#  RDD project -- GitHub repository governance setup
# =============================================================================
#  Applies every control described in the project's governance document:
#  rulesets on main, release and version tags; merge settings; security
#  features; Actions permissions; and collaborators.
#
#  Safe to re-run. Rulesets are matched by name and updated in place rather
#  than duplicated.
#
#  Usage
#  -----
#      bash .github/setup-repo.sh --repo OWNER/NAME                 # apply
#      bash .github/setup-repo.sh --repo OWNER/NAME --dry-run       # show only
#      bash .github/setup-repo.sh --repo OWNER/NAME --with-release  # + release branch
#      bash .github/setup-repo.sh --repo OWNER/NAME --require-ci    # + CI gate
#
#  --require-ci adds the CI workflow as a required status check. Run it ONLY
#  after the workflow has completed on at least one pull request; adding a
#  status check that has never reported blocks every merge.
# =============================================================================
set -euo pipefail
export MSYS_NO_PATHCONV=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=""
DRY=0
WITH_RELEASE=0
REQUIRE_CI=0
COLLABORATORS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --repo)         REPO="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    --with-release) WITH_RELEASE=1; shift ;;
    --require-ci)   REQUIRE_CI=1; shift ;;
    --collaborator) COLLABORATORS+=("$2"); shift 2 ;;
    -h|--help)      sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ -n "$REPO" ] || { echo "error: --repo OWNER/NAME is required" >&2; exit 2; }

say()  { printf '\n\033[1;36m### %s\033[0m\n' "$*"; }
ok()   { printf '  \033[0;32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[0;33mwarn\033[0m  %s\n' "$*"; }
skip() { printf '  \033[0;90mskip\033[0m  %s\n' "$*"; }

run() {
  if [ "$DRY" = "1" ]; then
    printf '  \033[0;90mwould run:\033[0m %s\n' "$*"
  else
    "$@"
  fi
}

# -----------------------------------------------------------------------------
say "Preflight"
# -----------------------------------------------------------------------------
command -v gh >/dev/null || { echo "gh CLI not found" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "not authenticated: run 'gh auth login'" >&2; exit 1; }
gh repo view "$REPO" >/dev/null 2>&1 || { echo "repository $REPO not found or not accessible" >&2; exit 1; }
ok "gh authenticated, $REPO reachable"

VISIBILITY=$(gh repo view "$REPO" --json visibility -q .visibility)
if [ "$VISIBILITY" != "PUBLIC" ]; then
  warn "repository is $VISIBILITY. Rulesets require a paid plan on private repos."
  warn "If ruleset creation fails below, that is why."
else
  ok "repository is public -- rulesets are available at no cost"
fi

# CODEOWNERS sanity. An entry naming a user who cannot be assigned as a reviewer
# is silently ignored, which leaves main unprotected while looking protected.
if [ -f "$HERE/CODEOWNERS" ]; then
  if grep -qE '^\*[[:space:]]+@' "$HERE/CODEOWNERS"; then
    ok "CODEOWNERS has a catch-all rule"
  else
    warn "CODEOWNERS has NO catch-all '*' rule -- paths not listed will not need review"
  fi
  if grep -q '@CHANGE-ME' "$HERE/CODEOWNERS"; then
    echo "error: CODEOWNERS still contains the @CHANGE-ME placeholder" >&2
    exit 1
  fi
else
  warn "no .github/CODEOWNERS found -- code-owner review will have no effect"
fi

# -----------------------------------------------------------------------------
say "Repository settings"
# -----------------------------------------------------------------------------
run gh api -X PATCH "repos/$REPO" \
  -F allow_merge_commit=false \
  -F allow_squash_merge=true \
  -F allow_rebase_merge=false \
  -F delete_branch_on_merge=true \
  -F allow_auto_merge=true \
  --silent
ok "squash-only merges, auto-delete merged branches, auto-merge enabled"

# -----------------------------------------------------------------------------
say "Security features"
# -----------------------------------------------------------------------------
run gh api -X PATCH "repos/$REPO" \
  -f 'security_and_analysis[secret_scanning][status]=enabled' \
  -f 'security_and_analysis[secret_scanning_push_protection][status]=enabled' \
  --silent 2>/dev/null \
  && ok "secret scanning + push protection enabled" \
  || warn "could not enable secret scanning (private repo, or already set)"

run gh api -X PUT "repos/$REPO/vulnerability-alerts" --silent 2>/dev/null \
  && ok "Dependabot alerts enabled" || warn "could not enable Dependabot alerts"

# Workflows get read-only tokens; nothing in this project's CI needs to write.
run gh api -X PUT "repos/$REPO/actions/permissions/workflow" \
  -f default_workflow_permissions=read \
  -F can_approve_pull_request_reviews=false \
  --silent 2>/dev/null \
  && ok "Actions token is read-only; workflows cannot approve PRs" \
  || warn "could not set Actions workflow permissions"

# -----------------------------------------------------------------------------
say "Rulesets"
# -----------------------------------------------------------------------------
# Read the "name" field without a JSON parser. `python` on Windows cannot open a
# Git Bash path like /c/Users/..., and jq is not always installed -- sed is the
# one tool guaranteed to be present wherever this script runs.
ruleset_name() {
  sed -n 's/^[[:space:]]*"name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$1" | head -1
}

# gh on Windows is a native .exe: a Git Bash path such as /c/Users/... or /tmp/...
# is meaningless to it, and `gh api --input` fails with "cannot find the path".
# Same fix as infra/config.sh uses for the AWS CLI.
nativepath() {
  if command -v cygpath >/dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

apply_ruleset() {
  local file="$1" name
  name=$(ruleset_name "$file")
  [ -n "$name" ] || { echo "error: no \"name\" field in $file" >&2; return 1; }

  local existing
  existing=$(gh api "repos/$REPO/rulesets" --jq \
      ".[] | select(.name == \"$name\") | .id" 2>/dev/null | head -1 || true)

  local native
  native=$(nativepath "$file")

  if [ -n "$existing" ]; then
    run gh api -X PUT "repos/$REPO/rulesets/$existing" --input "$native" --silent
    ok "updated ruleset '$name' (id $existing)"
  else
    run gh api -X POST "repos/$REPO/rulesets" --input "$native" --silent
    ok "created ruleset '$name'"
  fi
}

apply_ruleset "$HERE/rulesets/protect-main.json"
apply_ruleset "$HERE/rulesets/protect-tags.json"

if [ "$WITH_RELEASE" = "1" ]; then
  if git show-ref --verify --quiet refs/heads/release 2>/dev/null; then
    skip "release branch already exists locally"
  else
    run git branch release
    run git push -u origin release
  fi
  apply_ruleset "$HERE/rulesets/protect-release.json"
else
  skip "release ruleset (pass --with-release when you need a prod branch)"
fi

# -----------------------------------------------------------------------------
if [ "$REQUIRE_CI" = "1" ]; then
say "Required status check"
  # Insert the status-check rule into the existing rules array, without a JSON
  # parser: append it after the opening '"rules": [' line.
  TMP=$(mktemp)
  sed 's/^\([[:space:]]*\)"rules": \[$/\1"rules": [\n\1  { "type": "required_status_checks", "parameters": { "strict_required_status_checks_policy": true, "required_status_checks": [ { "context": "check" } ] } },/' \
      "$HERE/rulesets/protect-main.json" > "$TMP"
  grep -q 'required_status_checks' "$TMP" \
    || { echo "error: failed to inject the status-check rule" >&2; rm -f "$TMP"; exit 1; }
  apply_ruleset "$TMP"
  rm -f "$TMP"
  ok "CI 'check' job is now a required status check"
else
  skip "required status check -- re-run with --require-ci after CI has run once"
fi

# -----------------------------------------------------------------------------
if [ ${#COLLABORATORS[@]} -gt 0 ]; then
say "Collaborators"
  for u in "${COLLABORATORS[@]}"; do
    run gh api -X PUT "repos/$REPO/collaborators/$u" -f permission=push --silent
    ok "invited $u with Write access"
  done
  warn "Write only. Never grant Maintain or Admin -- both can edit rulesets."
fi

# -----------------------------------------------------------------------------
say "Result"
# -----------------------------------------------------------------------------
if [ "$DRY" = "1" ]; then
  echo "  dry run -- nothing was changed"
  exit 0
fi

gh api "repos/$REPO/rulesets" --jq '.[] | "  \(.name)  [\(.target)]  \(.enforcement)"' || true

cat <<'NEXT'

  Verify the controls actually fire -- an untested control is an assumed control:

      git switch main
      git commit --allow-empty -m "protection test"
      git push origin main          # EXPECT: rejected, GH006

  Note that the repository admin is on the bypass list, so as owner you WILL be
  able to push. Test from a collaborator's account, or temporarily remove the
  bypass, to confirm it blocks them.

NEXT
