import os
import sys


root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)
os.chdir(root)

FOOOCUS_SKIP_PREFLIGHT = os.environ.get("FOOOCUS_SKIP_PREFLIGHT", "").strip().lower() in ("1", "true", "yes", "on")
FORCE_JSON_ENV = os.environ.get("FOOOCUS_PREFLIGHT_JSON", "").strip().lower() in ("1", "true", "yes", "on")
QUIET_MODE = FORCE_JSON_ENV


def run_preflight_check(stage: str = "pre-update"):
    if FOOOCUS_SKIP_PREFLIGHT:
        if not QUIET_MODE:
            print(f"\n[Preflight] Stage '{stage}' skipped (FOOOCUS_SKIP_PREFLIGHT=1)\n")
        return None
    try:
        from modules.environment_preflight import run_preflight
        if not QUIET_MODE:
            print(f"\n[Preflight] Running {stage} environment check ...\n")
        report = run_preflight(
            root_dir=root,
            exit_on_error=False,
            print_report=True,
            use_colors=not QUIET_MODE,
            as_json=FORCE_JSON_ENV,
            stage=stage
        )
        return report
    except Exception as e:
        msg = f"\n[Preflight] Warning: Could not run preflight check ({e})\n"
        if FORCE_JSON_ENV:
            print(msg, file=sys.stderr)
        else:
            print(msg)
        return None


run_preflight_check(stage="pre-update")

try:
    import pygit2
    pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)

    repo = pygit2.Repository(os.path.abspath(os.path.dirname(__file__)))

    branch_name = repo.head.shorthand

    remote_name = 'origin'
    remote = repo.remotes[remote_name]

    remote.fetch()

    local_branch_ref = f'refs/heads/{branch_name}'
    local_branch = repo.lookup_reference(local_branch_ref)

    remote_reference = f'refs/remotes/{remote_name}/{branch_name}'
    remote_commit = repo.revparse_single(remote_reference)

    merge_result, _ = repo.merge_analysis(remote_commit.id)

    if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
        print("Already up-to-date")
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
        local_branch.set_target(remote_commit.id)
        repo.head.set_target(remote_commit.id)
        repo.checkout_tree(repo.get(remote_commit.id))
        repo.reset(local_branch.target, pygit2.GIT_RESET_HARD)
        print("Fast-forward merge")
    elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
        print("Update failed - Did you modify any file?")
except Exception as e:
    print('Update failed.')
    print(str(e))

print('Update succeeded.')
from launch import *
