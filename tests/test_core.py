import pytest

from src import config as C
from src.content_ranker import content_score, rank_by_content
from src.data_loader import (CatalogueError, Exercise, load_catalogue, make_profile,
                             validate_exercise_row, validate_profile)
from src.features import FEATURE_DIM, feature_vector
from src.linucb import LinUCB
from src.metrics import (constraint_violation_rate, cumulative_reward, ndcg_at_k,
                         precision_at_k)
from src import safety_filter as S


def prof(**kw):
    d = dict(age=30, bmi=24.0, goal="strength", fitness_level="beginner",
             equipment=["bodyweight"], restrictions=["none"])
    d.update(kw)
    return d


def ex(eid="E1", equipment=("bodyweight",), tags=("none",), status="reviewed",
       difficulty="beginner", goals=("strength",)):
    return Exercise(eid, eid, ("core",), tuple(equipment), difficulty, "strength",
                    tuple(goals), tuple(tags), "", status)


# --- profile validation (AC-01, AC-02)
def test_valid_profile():
    assert validate_profile(prof()) == []
    assert make_profile(prof()).restrictions == ()


@pytest.mark.parametrize("kw,field", [
    ({"age": 5}, "age"), ({"age": "x"}, "age"), ({"bmi": 1.0}, "bmi"),
    ({"goal": "x"}, "goal"), ({"fitness_level": None}, "fitness_level"),
    ({"equipment": []}, "equipment"), ({"equipment": ["jetpack"]}, "equipment"),
    ({"restrictions": None}, "restrictions"), ({"restrictions": ["none", "knee"]}, "restrictions"),
])
def test_invalid_profile_names_field(kw, field):
    errs = validate_profile(prof(**kw))
    assert errs and any(e.startswith(field) for e in errs)


# --- catalogue schema
def test_catalogue_loads_and_is_all_unreviewed():
    cat = load_catalogue()
    assert len(cat) >= 20
    assert all(e.review_status != "reviewed" for e in cat)


def test_unknown_equipment_fails_validation():
    row = dict(exercise_id="X", name="n", muscle_groups="core", equipment="rocket",
               difficulty="beginner", exercise_type="strength", goal_tags="strength",
               contraindication_tags="none")
    assert any("equipment" in e for e in validate_exercise_row(row))


def test_duplicate_ids_rejected(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("exercise_id,name,muscle_groups,equipment,difficulty,exercise_type,goal_tags,"
                 "contraindication_tags,instructions,review_status\n"
                 "A,a,core,bodyweight,beginner,strength,strength,none,,reviewed\n"
                 "A,b,core,bodyweight,beginner,strength,strength,none,,reviewed\n")
    with pytest.raises(CatalogueError):
        load_catalogue(p)


# --- safety rules (AC-03, 04, 05)
def test_equipment_excluded_with_reason():
    p = make_profile(prof())
    r = S.check_exercise(p, ex(equipment=("dumbbells",)))
    assert [x.reason_code for x in r] == [S.EQUIPMENT_UNAVAILABLE]


def test_bodyweight_always_available():
    p = make_profile(prof(equipment=["dumbbells"]))
    assert S.check_exercise(p, ex()) == []


def test_contraindication_conflict():
    p = make_profile(prof(restrictions=["knee"]))
    r = S.check_exercise(p, ex(tags=("knee", "hip")))
    assert [x.reason_code for x in r] == [S.CONTRAINDICATION_CONFLICT]
    assert S.check_exercise(p, ex(tags=("hip",))) == []


def test_missing_metadata_and_unreviewed_excluded():
    p = make_profile(prof())
    assert S.SAFETY_METADATA_MISSING in [x.reason_code for x in S.check_exercise(p, ex(tags=()))]
    assert S.NOT_REVIEWED in [x.reason_code for x in S.check_exercise(p, ex(status="draft_unreviewed"))]


def test_prefer_not_to_say_is_conservative():
    p = make_profile(prof(restrictions=["prefer_not_to_say"]))
    assert S.check_exercise(p, ex(tags=("none",))) == []
    assert S.check_exercise(p, ex(tags=("knee",)))


def test_filter_and_invariant():
    p = make_profile(prof(restrictions=["knee"]))
    good, bad = ex("G"), ex("B", tags=("knee",))
    eligible, excl = S.filter_exercises(p, [good, bad])
    assert eligible == [good] and excl[0].exercise_id == "B"
    S.assert_no_violations(p, eligible)
    with pytest.raises(AssertionError):
        S.assert_no_violations(p, [bad])


# --- content score
def test_content_score_range_and_order():
    p = make_profile(prof())
    s_hi, _ = content_score(p, ex("A"))
    s_lo, _ = content_score(p, ex("B", difficulty="advanced", goals=("endurance",)))
    assert 0 <= s_lo < s_hi <= 1
    assert rank_by_content(p, [ex("B", difficulty="advanced", goals=("endurance",)), ex("A")])[0][0].exercise_id == "A"


# --- LinUCB
def test_linucb_cold_start_keeps_prior_order_and_learns():
    p = make_profile(prof())
    a, b = ex("A", goals=("strength",)), ex("B", goals=("endurance",))
    xs = [feature_vector(p, a), feature_vector(p, b)]
    m = LinUCB(FEATURE_DIM)
    assert [c.exercise_id for c, _ in m.rank([a, b], xs, [0.9, 0.5])] == ["A", "B"]
    for _ in range(30):
        m.update(xs[1], C.REWARDS["completion"])
        m.update(xs[0], C.REWARDS["skip"])
    assert [c.exercise_id for c, _ in m.rank([a, b], xs, [0.5, 0.5])] == ["B", "A"]


def test_linucb_never_adds_items_and_reset_and_persist(tmp_path):
    p = make_profile(prof())
    a = ex("A")
    m = LinUCB(FEATURE_DIM)
    m.update(feature_vector(p, a), 1.0)
    f = tmp_path / "m.json"
    m.save(f)
    m2 = LinUCB.load(f)
    assert m2.n_updates == 1 and (m2.A == m.A).all()
    assert len(m2.rank([a], [feature_vector(p, a)])) == 1
    m2.reset()
    assert m2.n_updates == 0 and not m2.b.any()


# --- metrics
def test_metrics():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)
    assert ndcg_at_k(["a", "b"], {"a": 1, "b": 0}, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], {"a": 1, "b": 0}, 2) < 1.0
    assert ndcg_at_k(["a"], {}, 3) == 0.0
    assert constraint_violation_rate(1, 4) == 0.25 and constraint_violation_rate(0, 0) == 0.0
    assert cumulative_reward([1, -0.25, 0.5]) == [1, 0.75, 1.25]


def test_synthetic_users_are_reproducible_and_valid():
    from src.synthetic_users import generate
    a, b = generate(50, 1), generate(50, 1)
    assert a == b and generate(50, 2) != a
    for r in a:
        d = dict(age=r["age"], bmi=r["bmi"], goal=r["goal"], fitness_level=r["fitness_level"],
                 equipment=r["equipment"].split("|"), restrictions=r["restrictions"].split("|"))
        assert validate_profile(d) == []
