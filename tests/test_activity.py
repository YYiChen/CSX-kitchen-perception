import unittest

import helpers

from kitchen_perception.activity import ActivityObservation, ActivityStateMachine
from kitchen_perception.config import load_config
from kitchen_perception.motion import MotionFeatures


def make_chopping(ts_ms: float, detected: bool = True, stable: bool = True) -> MotionFeatures:
    return MotionFeatures(
        key="right#0",
        label="right",
        valid=True,
        source="palm",
        speed_norm_per_s=0.5,
        movement_level="HIGH",
        reversals=6 if detected else 0,
        rhythm_hz=2.0 if detected else 0.0,
        amplitude_ratio=0.12 if detected else 0.0,
        period_cv=0.1,
        amplitude_cv=0.1,
        score=0.8 if detected else 0.0,
        detected=detected,
        stable=stable and detected,
        last_chop_ms=ts_ms if detected else None,
    )


def make_observation(
    ts_ms: float,
    frame: int,
    person: bool = True,
    hands: bool = True,
    body: str = "A",
    chopping: MotionFeatures | None = None,
    movement: str = "LOW",
    stale: bool = False,
    gap_ms: float = 33.3,
) -> ActivityObservation:
    return ActivityObservation(
        ts_ms=ts_ms,
        frame_index=frame,
        person_visible=person,
        hands_valid=hands,
        hands_valid_count=2 if hands else 0,
        body_zone=body,
        chopping_zone=body if chopping is not None else "unknown",
        chopping=chopping,
        movement_level=movement,
        detection_gap_ms=gap_ms,
        stale=stale,
        camera_status="CONNECTED",
    )


class Harness:
    def __init__(self):
        self.machine = ActivityStateMachine(load_config())
        self.events: list[tuple[float, str, str]] = []
        self.states: list[tuple[float, str]] = []

    def feed(self, observation: ActivityObservation) -> dict:
        activity = self.machine.update(observation)
        self.states.append((observation.ts_ms, activity["state"]))
        for event in self.machine.take_events():
            self.events.append((observation.ts_ms, event.event, activity["state"]))
        return activity

    def names(self) -> list[str]:
        return [name for _, name, _ in self.events]

    def state_at(self, ts_ms: float) -> str:
        state = "UNKNOWN"
        for when, name in self.states:
            if when <= ts_ms:
                state = name
        return state

    def first_time(self, predicate) -> float | None:
        for ts, state in self.states:
            if predicate(state):
                return ts
        return None


class TestActivity(unittest.TestCase):
    def test_chop_starts_active_without_event_storm(self):
        h = Harness()
        for i in range(150):  # 5 s
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=make_chopping(ts), movement="HIGH"))
        self.assertEqual(h.machine.state, "ACTIVE_A")
        self.assertGreaterEqual(h.first_time(lambda s: s == "ACTIVE_A"), 500)
        self.assertLessEqual(h.first_time(lambda s: s == "ACTIVE_A"), 900)
        self.assertEqual(h.names().count("ACTIVITY_A_STARTED"), 1)
        self.assertEqual(h.names().count("CHOPPING_STARTED"), 1)
        self.assertNotIn("ACTIVITY_PAUSED", h.names())
        self.assertNotIn("HAND_OCCLUDED", h.names())

    def test_stop_enters_paused_not_away(self):
        h = Harness()
        for i in range(90):  # 3 s chop
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=make_chopping(ts), movement="HIGH"))
        for i in range(90, 180):  # 3 s still
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=None, movement="LOW"))
        self.assertEqual(h.machine.state, "PAUSED")
        self.assertIn("CHOPPING_STOPPED", h.names())
        self.assertIn("ACTIVITY_PAUSED", h.names())
        self.assertNotIn("USER_AWAY", h.names())

    def test_transition_a_to_b(self):
        h = Harness()
        for i in range(90):
            ts = i * 33.3
            h.feed(make_observation(ts, i, body="A", chopping=make_chopping(ts), movement="HIGH"))
        for i in range(90, 180):
            ts = i * 33.3
            h.feed(make_observation(ts, i, body="B", chopping=make_chopping(ts), movement="HIGH"))
        self.assertEqual(h.machine.state, "ACTIVE_B")
        self.assertEqual(h.names().count("ACTIVITY_A_STARTED"), 1)
        self.assertEqual(h.names().count("ZONE_TRANSITION_A_TO_B"), 1)
        self.assertEqual(h.names().count("ACTIVITY_B_STARTED"), 1)
        self.assertEqual(h.names().count("ACTIVITY_A_STARTED"), 1)

    def test_occlusion_not_paused(self):
        h = Harness()
        for i in range(30):  # 1 s chop
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=make_chopping(ts), movement="HIGH"))
        for i in range(30, 90):  # 2 s hands missing
            ts = i * 33.3
            h.feed(make_observation(ts, i, hands=False, chopping=None, movement="unknown"))
        self.assertEqual(h.machine.state, "OCCLUDED")
        self.assertIn("HAND_OCCLUDED", h.names())
        self.assertNotIn("ACTIVITY_PAUSED", h.names())
        self.assertEqual(h.machine.update(make_observation(89 * 33.3, 89, hands=False, chopping=None, movement="unknown"))["observation_quality"], "OCCLUDED")

    def test_away_then_return(self):
        h = Harness()
        for i in range(240):  # 8 s no person
            ts = i * 33.3
            h.feed(make_observation(ts, i, person=False, hands=False, body="unknown", movement="unknown"))
        self.assertEqual(h.machine.state, "AWAY")
        self.assertEqual(h.names().count("USER_AWAY"), 1)
        for i in range(240, 400):  # 5.3 s person visible, still
            ts = i * 33.3
            h.feed(make_observation(ts, i, person=True, hands=True, body="A", movement="LOW"))
        self.assertEqual(h.machine.state, "PAUSED")
        self.assertEqual(h.names().count("USER_RETURNED"), 1)

    def test_no_person_is_unknown_before_away(self):
        h = Harness()
        for i in range(60):  # 2 s，人不在画面但还没到 AWAY 阈值
            ts = i * 33.3
            h.feed(make_observation(ts, i, person=False, hands=False, body="unknown", movement="unknown"))
        self.assertEqual(h.machine.state, "UNKNOWN")
        self.assertNotIn("HAND_OCCLUDED", h.names())
        self.assertNotIn("USER_AWAY", h.names())

    def test_generic_movement_is_moving_not_active(self):
        h = Harness()
        for i in range(90):
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=None, movement="HIGH"))
        self.assertEqual(h.machine.state, "MOVING")
        self.assertNotIn("ACTIVITY_A_STARTED", h.names())
        self.assertIn("HAND_MOVING", h.names())

    def test_brief_gap_does_not_restart_active(self):
        h = Harness()
        for i in range(90):
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=make_chopping(ts), movement="HIGH"))
        for i in range(90, 96):  # 200 ms missing
            ts = i * 33.3
            h.feed(make_observation(ts, i, hands=False, chopping=None, movement="unknown"))
        for i in range(96, 180):
            ts = i * 33.3
            h.feed(make_observation(ts, i, chopping=make_chopping(ts), movement="HIGH"))
        self.assertEqual(h.machine.state, "ACTIVE_A")
        self.assertEqual(h.names().count("ACTIVITY_A_STARTED"), 1)
        self.assertEqual(h.names().count("CHOPPING_STARTED"), 1)

    def test_degraded_sampling_still_detects(self):
        h = Harness()
        for i in range(60):  # 15 FPS，模拟 L1/L3 隔帧
            ts = i * 66.6
            activity = h.feed(
                make_observation(ts, i, chopping=make_chopping(ts), movement="HIGH", gap_ms=66.6)
            )
        self.assertEqual(h.machine.state, "ACTIVE_A")
        self.assertIn(activity["observation_quality"], ("VALID", "LOW"))

    def test_no_person_quality(self):
        h = Harness()
        activity = h.feed(make_observation(0.0, 0, person=False, hands=False, body="unknown", movement="unknown"))
        self.assertEqual(activity["observation_quality"], "NO_PERSON")
        self.assertEqual(activity["state"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
