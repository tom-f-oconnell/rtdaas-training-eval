from __future__ import annotations

import os

import httpx
import pytest

from training_eval.client import ApiClient
from training_eval.dataset import build_table


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/trials":
        trials = [{"trial_id": "Suturing_B001", "task": "Suturing", "subject": "B",
                   "repetition": 1, "skill_level": "novice", "grs_total": 13}]  # fmt: skip
        return httpx.Response(200, json={"trials": trials})
    if path == "/api/trials/Suturing_B001":
        return httpx.Response(200, json={"gestures": [{"gesture": "G1", "duration_s": 3.0}]})
    if path.endswith("/kinematics"):
        assert request.url.params["arms"] == "psml,psmr,mtml,mtmr"
        row = {"frame_idx": 0, "t_s": 0.0, "gesture": "G1"}
        for arm in ("psml", "psmr", "mtml", "mtmr"):
            row |= {f"{arm}_{k}": 0.0 for k in ("x", "y", "z", "wx", "wy", "wz", "gripper")}
        rows = [
            {**row, "frame_idx": i, "psml_x": i * 0.001, "mtml_x": i * 0.003} for i in range(200)
        ]
        return httpx.Response(200, json={"samples": rows})
    if path.endswith("/frames"):
        return httpx.Response(200, json={"frames": []})
    return httpx.Response(404, json={"error": "not_found"})


def _client():
    return ApiClient("http://api.test", transport=httpx.MockTransport(_handler))


def test_build_table_reads_the_routes_and_caches(tmp_path):
    table = build_table(_client(), cache_dir=tmp_path)
    assert list(table["trial_id"]) == ["Suturing_B001"]
    assert table.loc[0, "gesture_distinct"] == 1
    assert (tmp_path / "Suturing_B001.pkl.gz").exists()

    def no_network(request):
        if request.url.path != "/api/trials":
            pytest.fail(f"cache missed: {request.url.path}")
        return _handler(request)

    cached = ApiClient("http://api.test", transport=httpx.MockTransport(no_network))
    assert list(build_table(cached, cache_dir=tmp_path)["trial_id"]) == ["Suturing_B001"]


def test_an_empty_api_is_a_clear_error():
    empty = httpx.MockTransport(lambda r: httpx.Response(200, json={"trials": []}))
    with pytest.raises(RuntimeError, match="has no matching trials"):
        build_table(ApiClient("http://api.test", transport=empty))


def test_task_filter():
    with pytest.raises(RuntimeError):
        build_table(_client(), tasks=["Knot_Tying"])


def test_api_errors_are_raised():
    with pytest.raises(httpx.HTTPStatusError):
        _client().trial("missing")


@pytest.mark.integration
@pytest.mark.skipif("RTDAAS_API_URL" not in os.environ, reason="needs RTDAAS_API_URL")
def test_against_a_running_api():
    client = ApiClient()
    meta = client.trials()[0]
    assert {"trial_id", "task", "subject", "repetition", "skill_level"} <= set(meta)
    kinematics = client.kinematics(meta["trial_id"])
    assert len(kinematics) > 100 and "mtmr_gripper" in kinematics
