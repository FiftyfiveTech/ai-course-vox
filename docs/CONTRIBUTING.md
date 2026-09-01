# Contributing — branches, pull requests, merges

Everything lands through a pull request reviewed by the other developer. This file is the
whole procedure: what to run before you open one, how to open it, how to review one, how to
merge it, and how to get out of the three situations that go wrong most often.

If you only read one paragraph: cut a branch off `dev`, keep commits small and green, open
the PR against `dev`, request the *other* person as reviewer, and **let them press merge**.

---

## Branch model

| Branch | What it is | Who writes to it |
|---|---|---|
| `main` | Milestone/handoff branch. Protected. | Only a reviewed `dev` → `main` PR |
| `dev` | Integration branch — the branch every ticket targets | Only merged PRs |
| `feat/vox-NNN-<slug>` | One ticket, one branch | You |

Other prefixes exist for work that is not a numbered ticket: `fix/…` for a bug,
`enhancement/…` for an unticketed improvement, `docs/…` for documentation. Same rules apply
to all of them.

**Nothing is committed directly to `dev` or `main`.** Not a typo fix, not a one-line README
change. The Friday retro reads `git log` for exactly this.

---

## 1 · Start a branch

```bash
git checkout dev
git pull --ff-only origin dev            # always pull before you branch
git checkout -b feat/vox-014-endpointing
```

`--ff-only` on purpose: if it refuses, your local `dev` has commits of its own, which is
itself the bug. Sort that out before writing code, not at merge time.

One ticket per branch. If a second ticket turns out to be in the way, stop and say so —
do not widen the branch.

---

## 2 · Commit as you go

Conventional Commits, with the ticket id in the subject:

```
<type>(vox-NNN): <imperative summary>

Why the change is needed. What you deviated from in the ticket spec, and why.
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`. Examples in this repo's style:

```
feat(vox-011): stop the reply when the user talks over it
fix(vox-007): fall back to the local arm on 429 rather than losing the turn
docs(vox-003): record the measured latency split and why the budget misses
```

Rules that make review possible:

- **Each commit is one logical change and leaves `make test` green.** Split by layer if it
  helps: config/types → implementation → tests → docs.
- No `WIP` / `fixes` / `asdf` commits. Squash them locally before pushing.
- **Never commit** `runs/`, `.env`, `*.wav`, `*.mp4`, `docs/learning/coach/chat.jsonl`, or
  anything under `evals/heldout/` that the Builder is not allowed to read. Most of this is
  already git-ignored — do not `git add -f` past it.
- **No AI attribution anywhere in git or GitHub artifacts** — no co-author trailer naming an
  AI model, no `@anthropic.com` address, no "generated with / assisted by" footer, in
  commits, PR titles, PR bodies or review comments. The developers are the authors. If a tool
  appends such a footer, strip it before committing.
- A measured number quoted in a commit or PR body must come with the command that produced
  it. No remembered numbers.

---

## 3 · Before you open the PR

Run these, and paste the real output into the PR body:

```bash
make test                     # unit suite — must be green
make gate                     # phase gates, if the ticket has one
git status --short            # nothing stray, nothing ignored force-added
git log --oneline dev..HEAD   # the story a reviewer is about to read
```

Then check the ticket's own acceptance criterion. It is the ticket description, not your
reading of it — re-read it before claiming the PR is done.

If the ticket doubles as course material (it usually does), the primer
`docs/learning/vox-<nnn>-concepts.md` and the `docs/learning/retros.md` entry are part of
*this* PR. They are deliverables, not scratch — see
[docs/learning/README.md](learning/README.md).

---

## 4 · Open the PR

```bash
git push -u origin feat/vox-014-endpointing

gh pr create \
  --base dev \
  --head feat/vox-014-endpointing \
  --title "feat(vox-014): tune endpointing on the dev set" \
  --reviewer <the-other-developer> \
  --body-file /tmp/pr-body.md      # or --body "…", or --web to write it in the browser
```

`--base dev` is not optional — a PR that targets `main` by accident is the single most
common mistake here, and GitHub will happily let you merge it.

**PR body template** (keep it short, keep it honest):

```markdown
## What
One or two sentences. What now works that did not before.

## Ticket
VOX-NNN — <acceptance criterion, quoted from the ticket>

## Numbers
$ make gate
<paste the real output — the number decides, not the description>

## Deviations
Where this differs from the ticket spec and why. "None" is a valid answer.

## Review notes
Anything you want a second pair of eyes on, and anything you know is still weak.
```

Open it as a **draft** (`--draft`) if you want early eyes without asking for a merge; mark it
ready with `gh pr ready` when it is.

---

## 5 · Review someone else's PR

The reviewer's job is not to approve — it is to be able to explain the change afterwards.

```bash
gh pr list                         # what is waiting
gh pr view 14                      # description, checks, files
gh pr diff 14                      # read the diff
gh pr checkout 14                  # run it yourself
make test && make gate             # re-run the numbers; claimed ≠ verified
```

Check, in this order:

1. **Does the number reproduce on your machine?** A gate that only passes on the author's
   laptop has not passed.
2. **Does the code match the ticket's acceptance criterion**, including the parts that were
   inconvenient?
3. **Deviations declared?** A deviation is fine; an undeclared one is not.
4. **Anything committed that should not be** — secrets, `runs/`, media, held-out labels, an
   AI attribution footer.
5. **Can you explain it?** If not, ask in the PR until you can. That is the whole point of
   the review, and the reason nobody merges their own work.

Then:

```bash
gh pr review 14 --approve -b "Reproduced make gate locally: 0.83. Reads clean."
gh pr review 14 --request-changes -b "gate fails here at 0.71 — see comment on src/vad.py:88"
gh pr review 14 --comment -b "Question about the cooldown default before I approve"
```

---

## 6 · Merge

**The reviewer merges, never the author.** `CLAUDE.md` states it as a hard rule and the
Friday retro checks `git log` for it — and the recent history of this repo shows the rule
being broken more often than kept, so it is worth restating: if your name is on the commits,
your name does not go on the merge.

```bash
gh pr merge 14 --merge --delete-branch      # merge commit, then delete the remote branch
```

`--merge` (a merge commit) is what this repo has used for every feature → `dev` PR so far, and
keeping one strategy makes the history readable. Use `--squash` only for a branch whose
individual commits are noise, and say so in the PR.

If the merge button is greyed out:

- **"This branch is out-of-date with the base branch"** → the author updates the branch
  (§7), not the reviewer.
- **Conflicts** → the author resolves them (§7).
- **Checks failing** → not a merge problem. Fix the failure.

### After the merge

```bash
git checkout dev
git pull --ff-only origin dev
git branch -d feat/vox-014-endpointing     # local cleanup; the remote one is already gone
```

Then close the loop outside git: comment the result on the Odoo ticket (what was built,
deviations, test output, files touched), append the retro entry if it was not in the PR, and
add today's line to `STANDUP.md`.

### Promoting `dev` → `main`

Same procedure, one level up: `gh pr create --base main --head dev --title "release: <what>"`,
reviewed by the other developer, merged with `--merge`. `main` is the branch someone else
clones cold, so a `dev` → `main` PR means `make setup && make test` has been run from a fresh
clone, not just from your working copy.

---

## 7 · Keeping the branch current

`dev` moves while you work. Bring it in — do not let the branch drift for days.

```bash
git checkout feat/vox-014-endpointing
git fetch origin
git merge origin/dev              # merge, matching how this repo integrates
# resolve conflicts, then:
make test
git push
```

Rebase (`git rebase origin/dev`) is fine on a branch **nobody else has checked out**, because
it rewrites history and forces `git push --force-with-lease`. Never rebase or force-push a
branch someone is reviewing — their comments detach from the commits and the review is lost.
`--force-with-lease`, never bare `--force`: it refuses if someone else pushed in the
meantime, which is exactly the case you want to be stopped in.

---

## 8 · When it goes wrong

**Committed a secret.** Stop. Rotate the credential first — assume it is public the moment
it was pushed. Then remove it from the branch (`git rebase -i` on an unshared branch, or a
fresh branch with clean commits) and force-push with `--force-with-lease`. Rewriting shared
history is a conversation with the other developer, not a solo decision.

**Committed to `dev` by accident, not yet pushed.**

```bash
git branch feat/vox-014-endpointing      # save the work first
git reset --hard origin/dev              # DESTRUCTIVE: discards uncommitted changes in the
                                         # working tree. Run `git status` first and stash
                                         # anything you still need.
git checkout feat/vox-014-endpointing
```

**Pushed to `dev` by accident.** Do not force-push over a shared branch. Open a revert PR
(`gh pr create` from a branch containing `git revert <sha>`) and re-land the work properly.

**PR opened against the wrong base.** No need to close it: `gh pr edit 14 --base dev`.

**Reviewer unavailable and the work is blocking.** Say so on the ticket and wait, or ask the
supervisor to review. "It was blocking" is not a licence to self-merge.

---

## Checklists

**Author, before requesting review**

- [ ] Branch cut from an up-to-date `dev`, named `feat/vox-NNN-<slug>`
- [ ] `make test` green; `make gate` output pasted in the PR body
- [ ] Commits are Conventional, one logical change each, no WIP noise
- [ ] No secrets, `runs/`, media, held-out labels, or AI attribution
- [ ] Primer + retro committed if the ticket doubles as course material
- [ ] PR base is `dev`; the other developer is requested as reviewer

**Reviewer, before approving**

- [ ] Checked the branch out and re-ran the numbers yourself
- [ ] Acceptance criterion actually met, deviations declared
- [ ] Nothing committed that should not be
- [ ] You can explain the change without the author in the room
- [ ] You are not the author (if you are: stop, get the other person)
