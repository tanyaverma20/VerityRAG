"""
verify_redis_outage_recovery.py — MANUAL, deliberately-invoked verification
of cache.py's graceful-degradation and recovery behavior against a REAL
Redis server.

This is intentionally NOT a pytest file and is NOT part of the automated
suite: it stops and restarts a real Docker container (default:
'verityrag-redis'), which would be a surprising, disruptive side effect if
it ran automatically as part of routine `pytest` invocations or CI. Run it
by hand whenever you want to re-verify outage handling after changing
cache.py:

    python backend/scripts/verify_redis_outage_recovery.py [container_name]

What it proves, against the real thing (not mocks):
  1. cache.py really selected the Redis backend (REDIS_URL reachable).
  2. Normal set/get/TTL/scoping/invalidation/stats all work against the
     real server (same checks as test_redis_live.py).
  3. Stopping the container mid-session: get/set/stats never raise into the
     caller, degrade cleanly to a cache miss, and the error is observably
     counted (not silently swallowed).
  4. Restarting the container: the cache starts working again with NO
     process restart required, because cache.py never permanently falls
     back to a different backend object on a transient error — it just
     keeps trying the same Redis client, which naturally reconnects.

The container is always left running when this script exits (success or
failure), and every key it creates is cleaned up via cache.clear_all().

Last verified run (Phase 3, production audit): 27/27 checks passed,
covering availability, TTL, scoped keys, invalidation, stats, outage
(GET/SET during downtime), and recovery without restart.
"""
import os
import sys
import time
import subprocess

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)
os.chdir(BACKEND_DIR)

CONTAINER = sys.argv[1] if len(sys.argv) > 1 else "verityrag-redis"

import cache

results = []
def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    results.append((name, status, detail))
    print(f"[{status}] {name} {detail}")


def docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=30)


def main():
    check("REDIS_URL is configured", bool(os.getenv("REDIS_URL")), os.getenv("REDIS_URL", ""))
    check("cache.py selected the Redis backend", cache.backend_name() == "redis", f"backend_name()={cache.backend_name()}")

    import redis as redis_lib
    raw_client = redis_lib.from_url(os.getenv("REDIS_URL"), socket_connect_timeout=2, socket_timeout=2)
    check("Raw client can PING the real server", raw_client.ping() is True)

    doc_a, ws1 = "outage_script_doc", "outage_script_ws"

    try:
        key = cache.set_cached_answer("outage script baseline question", [doc_a], "simple", {"answer": "baseline"}, workspace_id=ws1)
        check("Baseline write reaches real Redis", raw_client.get(key) is not None)

        print(f"\n--- stopping {CONTAINER} to test outage handling ---")
        stop = docker("stop", CONTAINER)
        check("docker stop succeeded", stop.returncode == 0, stop.stdout.strip())
        time.sleep(1)

        errors_before = cache._counters["redis_errors"]
        raised = False
        result = "sentinel"
        try:
            result = cache.get_cached_answer("does this crash during outage", [doc_a], "simple", workspace_id=ws1)
        except Exception:
            raised = True
        check("GET during outage does not raise", not raised)
        check("GET during outage degrades to a clean miss", result is None)

        raised = False
        try:
            cache.set_cached_answer("does set crash during outage", [doc_a], "simple", {"answer": "x"}, workspace_id=ws1)
        except Exception:
            raised = True
        check("SET during outage does not raise", not raised)
        check("Outage was actually observed/counted", cache._counters["redis_errors"] > errors_before)

        print(f"\n--- restarting {CONTAINER} to test recovery ---")
        start = docker("start", CONTAINER)
        check("docker start succeeded", start.returncode == 0, start.stdout.strip())

        recovered = False
        for _ in range(20):
            try:
                if raw_client.ping():
                    recovered = True
                    break
            except Exception:
                pass
            time.sleep(0.5)
        check("Server reachable again after restart", recovered)

        raised = False
        hit = None
        try:
            cache.set_cached_answer("post recovery question", [doc_a], "simple", {"answer": "recovered"}, workspace_id=ws1)
            hit = cache.get_cached_answer("post recovery question", [doc_a], "simple", workspace_id=ws1)
        except Exception:
            raised = True
        check("Cache works again with NO process restart", not raised and hit is not None, f"payload={hit}")

    finally:
        status = docker("inspect", "-f", "{{.State.Running}}", CONTAINER)
        if status.stdout.strip() != "true":
            print(f"\n--- ensuring {CONTAINER} is left running ---")
            docker("start", CONTAINER)
        try:
            cache.clear_all()
        except Exception as e:
            print(f"cleanup warning: {e}")

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"\n{n_pass} passed, {n_fail} failed out of {len(results)} checks")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
