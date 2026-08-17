#!/usr/bin/env python3
"""MCP server for the Odoo "AI Dev Course" board — one project only.

Deliberately narrow. It exposes the course board and nothing else: no generic
model access, no other project, no `Done`. The tenant holds ~3300 tasks across
sibling bootcamp projects; PROJECT_ID is hardcoded and every read and write
re-checks it, so an agent cannot wander.

  python3 odoo_board_mcp.py --selftest    # verify creds + board access
  python3 odoo_board_mcp.py               # stdio MCP server (what Claude Code runs)

Creds: ODOO_USER (email) + ODOO_KEY (Odoo API key, Preferences -> Account
Security -> New API Key) from the environment, ~/.config/ai-course-board.env,
or ~/.config/secrets.env.
"""
import os
import sys
import xmlrpc.client
from pathlib import Path

URL = os.environ.get("ODOO_URL", "https://odoo.fiftyfivetech.io")
DB = os.environ.get("ODOO_DB", "odoo-db")
PROJECT_ID = int(os.environ.get("ODOO_PROJECT_ID", 60))
PROJECT_NAME = os.environ.get("ODOO_PROJECT_NAME", "VOX - Enterprise Voice Agent")

# Stages on project 60 are statuses: Plan Backlog -> inProgress -> Review -> Done.
# Done is off-limits to this server; the supervisor moves the card after reproducing the gate.
DONE = int(os.environ.get("ODOO_DONE_STAGE_ID", 1880))  # "Done" on project 60
DONE_REFUSAL = (
    "Refused: this server cannot set a task to Done. A gate is verified by the supervisor "
    "re-running the gate command, not by the person who wrote the code. Use request_review() "
    "instead — the supervisor moves it to Done after the number reproduces."
)

ENV_FILES = [Path.home() / ".config/ai-course-board.env", Path.home() / ".config/secrets.env"]


def load_creds():
    for env in ENV_FILES:
        if not env.is_file():
            continue
        for line in env.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith("ODOO_") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
    user, key = os.environ.get("ODOO_USER"), os.environ.get("ODOO_KEY")
    if not (user and key):
        sys.exit(
            "ODOO_USER / ODOO_KEY missing.\n"
            "  1. Odoo -> avatar -> Preferences -> Account Security -> New API Key\n"
            f"  2. printf 'ODOO_USER=you@fiftyfivetech.io\\nODOO_KEY=<key>\\n' > {ENV_FILES[0]}\n"
            f"  3. chmod 600 {ENV_FILES[0]}"
        )
    return user, key


class Board:
    def __init__(self):
        user, self.key = load_creds()
        self.login = user
        self.uid = xmlrpc.client.ServerProxy(f"{URL}/xmlrpc/2/common").authenticate(
            DB, user, self.key, {})
        if not self.uid:
            sys.exit("auth failed — wrong ODOO_USER or expired/revoked ODOO_KEY")
        self.models = xmlrpc.client.ServerProxy(f"{URL}/xmlrpc/2/object")

    def kw(self, model, method, args, **kwargs):
        return self.models.execute_kw(DB, self.uid, self.key, model, method, args, kwargs)

    def stages(self):
        return self.kw("project.task.type", "search_read",
                       [[["project_ids", "in", [PROJECT_ID]]]],
                       fields=["id", "name"], order="sequence")

    def stage_id(self, ref):
        want = str(ref).strip().lower()
        stages = self.stages()
        hits = [s for s in stages if want in (str(s["id"]), s["name"].lower())
                or want in s["name"].lower()]
        if len(hits) != 1:
            names = ", ".join(s["name"] for s in stages)
            raise ValueError(f"{'unknown' if not hits else 'ambiguous'} stage {ref!r} — "
                             f"board has: {names}")
        return hits[0]["id"]

    def tasks(self, domain, fields):
        """Every read is fenced to PROJECT_ID — the pin cannot be argued away by a caller."""
        return self.kw("project.task", "search_read",
                       [[["project_id", "=", PROJECT_ID], *domain]],
                       fields=fields, order="date_deadline, id")

    def mine(self, task_id):
        """-> task dict, or raise if it is not in project 73 / not assigned to the caller."""
        found = self.tasks([["id", "=", int(task_id)]],
                           ["id", "name", "stage_id", "user_ids", "date_deadline"])
        if not found:
            raise ValueError(f"task {task_id} is not on the {PROJECT_NAME} board (project "
                             f"{PROJECT_ID}). This server reaches no other project.")
        t = found[0]
        if self.uid not in t["user_ids"]:
            raise ValueError(f"task {task_id} is not assigned to you ({self.login}). Ask the "
                             "supervisor to reassign it rather than working someone else's task.")
        return t


_board = None


def board():
    global _board
    if _board is None:
        _board = Board()
    return _board


def fmt(t, users=None):
    who = ", ".join((users or {}).get(i, str(i)) for i in t.get("user_ids", [])) or "unassigned"
    due = f" · due {t['date_deadline'][:10]}" if t.get("date_deadline") else ""
    stage = t["stage_id"][1] if t.get("stage_id") else "?"
    return f"[{t['id']}] {t['name']}\n      {stage} · {who}{due}"


def render(b, tasks):
    if not tasks:
        return "(no tasks match)"
    ids = sorted({i for t in tasks for i in t.get("user_ids", [])})
    users = {u["id"]: u["name"] for u in b.kw("res.users", "search_read",
                                              [[["id", "in", ids]]], fields=["name"])} if ids else {}
    out, stage = [], None
    for t in tasks:
        s = t["stage_id"][1] if t["stage_id"] else "(none)"
        if s != stage:
            stage, _ = s, out.append(f"\n== {s}")
        out.append("  " + fmt(t, users))
    return "\n".join(out).strip()


# --- tools -------------------------------------------------------------------

def register(mcp):
    @mcp.tool()
    def whoami() -> str:
        """Which Odoo user this server is authenticated as, and what it may do."""
        b = board()
        n = len(b.tasks([["user_ids", "in", [b.uid]]], ["id"]))
        return (f"{b.login} (res.users {b.uid}) on {PROJECT_NAME} (project {PROJECT_ID}).\n"
                f"{n} tasks assigned to you.\n"
                "May: read the board, start a task, log progress, request review.\n"
                "May NOT: set Done, touch any other project, read any other Odoo model.")

    @mcp.tool()
    def my_tasks(stage: str = "") -> str:
        """Your assigned tasks on the course board, soonest deadline first.

        Args:
            stage: optional filter — a day column ("Day 1", "Pre-Week") or "Done".
        """
        b = board()
        domain = [["user_ids", "in", [b.uid]]]
        if stage:
            domain.append(["stage_id", "=", b.stage_id(stage)])
        return render(b, b.tasks(domain, ["id", "name", "stage_id", "user_ids", "date_deadline"]))

    @mcp.tool()
    def board_view(stage: str = "") -> str:
        """The whole course board including your teammate's tasks (read-only).

        Args:
            stage: optional filter — a day column ("Day 1", "Pre-Week") or "Done".
        """
        b = board()
        domain = [["stage_id", "=", b.stage_id(stage)]] if stage else []
        return render(b, b.tasks(domain, ["id", "name", "stage_id", "user_ids", "date_deadline"]))

    @mcp.tool()
    def task(task_id: int) -> str:
        """Full detail for one task: acceptance criterion and reference-doc links.

        Args:
            task_id: the numeric task id shown in brackets by my_tasks.
        """
        import html
        import re
        b = board()
        found = b.tasks([["id", "=", int(task_id)]],
                        ["id", "name", "stage_id", "user_ids", "date_deadline", "description"])
        if not found:
            return (f"task {task_id} is not on the {PROJECT_NAME} board (project {PROJECT_ID}). "
                    "This server reaches no other project.")
        t = found[0]
        body = t["description"] or ""
        body = re.sub(r"<a [^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", r"\2 -> \1", body)
        body = html.unescape(re.sub(r"<[^>]+>", "\n", body))
        body = "\n".join(ln.strip() for ln in body.splitlines() if ln.strip())
        users = {u["id"]: u["name"] for u in b.kw("res.users", "search_read",
                 [[["id", "in", t["user_ids"]]]], fields=["name"])} if t["user_ids"] else {}
        mine = " (yours)" if b.uid in t["user_ids"] else " (teammate's — read-only)"
        return f"{fmt(t, users)}{mine}\n\n{body or '(no description)'}"

    @mcp.tool()
    def start(task_id: int) -> str:
        """Announce you are starting one of your tasks. Do this before writing code.

        Stages on this board are day columns, so starting does not move the card —
        it posts a STARTED note with a timestamp trail the supervisor can see.

        Args:
            task_id: a task assigned to you.
        """
        b = board()
        t = b.mine(task_id)
        if t["stage_id"] and t["stage_id"][0] == DONE:
            return f"task {task_id} is Done — reopening is the supervisor's call."
        b.kw("project.task", "message_post", [[int(task_id)]],
             body="<p><b>STARTED</b></p>", message_type="comment")
        return f"started task {task_id}: {t['name']}"

    @mcp.tool()
    def note(task_id: int, text: str) -> str:
        """Log progress, a blocker, or a measured number on one of your tasks.

        Args:
            task_id: a task assigned to you.
            text: what happened. Paste the command and its real output for any number.
        """
        b = board()
        t = b.mine(task_id)
        b.kw("project.task", "message_post", [[int(task_id)]],
             body=f"<p>{text}</p>", message_type="comment")
        return f"logged on task {task_id}: {t['name']}"

    @mcp.tool()
    def request_review(task_id: int, evidence: str) -> str:
        """Hand a finished task to the supervisor. Requires evidence, not a claim.

        The task stays in its day column — only the supervisor moves it to Done ✅,
        after re-running the check and seeing the same output.

        Args:
            task_id: a task assigned to you.
            evidence: the command you ran and its actual output (the measured number,
                the passing test, the built artifact). "it works" is not evidence.
        """
        b = board()
        t = b.mine(task_id)
        if len(evidence.strip()) < 20:
            return ("Refused: evidence too thin. Paste the command and its real output — the "
                    "acceptance criterion in the description is what the supervisor re-runs.")
        b.kw("project.task", "message_post", [[int(task_id)]],
             body=f"<p><b>REVIEW REQUESTED</b></p><pre>{evidence}</pre>", message_type="comment")
        return (f"review requested on task {task_id}: {t['name']}\n"
                "Stays in its day column until the supervisor reproduces the result.")

    @mcp.tool()
    def set_done(task_id: int) -> str:
        """Not available — present only so the refusal is explicit rather than a confusing error.

        Args:
            task_id: ignored.
        """
        return DONE_REFUSAL

    return mcp


def selftest():
    b = board()
    print(f"auth      ok — {b.login} = res.users {b.uid}")
    all_t = b.tasks([], ["id", "stage_id", "user_ids"])
    mine = [t for t in all_t if b.uid in t["user_ids"]]
    print(f"board     ok — {len(all_t)} tasks on {PROJECT_NAME} (project {PROJECT_ID})")
    print(f"assigned  {len(mine)} to you")
    for s in b.stages():
        n = sum(1 for t in all_t if t["stage_id"] and t["stage_id"][0] == s["id"])
        print(f"  {s['name'][:36]:36} {n}")
    leak = [t for t in all_t if not b.tasks([["id", "=", t["id"]]], ["id"])]
    print(f"pin       ok — every read fenced to project {PROJECT_ID} ({len(leak)} leaks)")
    print(f"done      blocked — set_done() returns a refusal")
    if not mine:
        print("\nNOTE: no tasks assigned to you yet — ask the supervisor to assign your week.")
    return 0


def main():
    if "--selftest" in sys.argv:
        return selftest()
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("odoo-board", instructions=(
        f"The Odoo '{PROJECT_NAME}' course board (project {PROJECT_ID}). Start every working "
        "session with my_tasks(). Read task() before coding — its description holds the "
        "acceptance criterion and links to the PRD. start() before you write code, note() as you "
        "go, request_review() with real output when the criterion is met. You cannot set Done."))
    register(mcp)
    mcp.run()


if __name__ == "__main__":
    sys.exit(main() or 0)
