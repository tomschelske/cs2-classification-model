"""Smoke tests over the deployed API surface (needs only the serving deps)."""


def test_health(client):
    d = client.get("/health").json()
    assert d["status"] == "ok"
    assert d["round_loaded"] is True
    assert d["fast_kernel"] is True          # the folded linear kernel is active


def test_predict_returns_valid_probability(client):
    r = client.post("/predict", json={})     # empty body -> defaults
    assert r.status_code == 200
    p = r.json()["t_win_prob"]
    assert 0.0 <= p <= 1.0


def test_partial_body_uses_defaults(client):
    # the calculator sends only the controls it exposes; the rest ride on defaults
    r = client.post("/predict", json={"equip_value_t": 800, "equip_value_ct": 20000})
    assert r.status_code == 200


def test_economy_is_the_dominant_signal(client):
    eco = client.post("/predict", json={"equip_value_t": 800, "equip_value_ct": 20000}).json()["t_win_prob"]
    full = client.post("/predict", json={"equip_value_t": 20000, "equip_value_ct": 800}).json()["t_win_prob"]
    assert eco < 0.3 and full > 0.7          # flipping the buy swings the prediction hard
    assert full - eco > 0.4


def test_man_advantage_extremes(client):
    up = client.post("/predict", json={"players_alive_ct": 0, "total_health_ct": 0}).json()["t_win_prob"]
    down = client.post("/predict", json={"players_alive_t": 0, "total_health_t": 0}).json()["t_win_prob"]
    assert up > 0.9                          # 5v0 -> T almost certainly wins
    assert down < 0.1                        # 0v5 -> T almost certainly loses


def test_validation_rejects_out_of_range(client):
    r = client.post("/predict", json={"players_alive_t": 6})   # field is le=5
    assert r.status_code == 422


def test_round_endpoint_structure(client):
    d = client.get("/round").json()
    assert {"meta", "series", "events"} <= set(d)
    assert d["meta"]["map"] == "de_dust2" and d["meta"]["round"] == 8
    assert len(d["series"]) > 0
    assert all({"t", "p"} <= set(s) for s in d["series"])
    kills = [e for e in d["events"] if e["type"] == "kill"]
    assert kills, "expected kill events"
    scored = [e for e in kills if e["leverage"]]
    assert scored, "expected leverage on at least some kills"
    assert "swing_pp" in scored[0]["leverage"]


def test_dashboard_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Live Dashboard" in r.text
