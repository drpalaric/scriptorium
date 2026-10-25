package scriptorium.change_control_test

import data.scriptorium.change_control as cc

ev(seq, entity, event, actor, payload) := {
	"seq": seq,
	"ts": "2026-10-25T00:00:00+00:00",
	"kind": "feedback",
	"entity_id": entity,
	"event": event,
	"actor": actor,
	"payload": payload,
}

good_log := [
	ev(1, "fb_1", "submitted", "user:alice", {}),
	ev(2, "fb_1", "approved", "admin:phil", {"reason": "ok"}),
	ev(3, "fb_1", "applied", "admin:phil", {"gate": {"passed": true}, "forced": false, "reason": null}),
]

ok_chain := {"ok": true}

rules_hit(r) := {v.rule | some v in r.violations}

run(events) := r if {
	r := cc.report with input as {"events": events, "chain": ok_chain}
}

test_a_clean_log_has_no_violations_or_exceptions if {
	r := run(good_log)
	count(r.violations) == 0
	count(r.exceptions) == 0
	r.events_checked == 3
}

test_every_violation_carries_a_rule_id_and_nist_controls if {
	bad := [
		ev(1, "fb_2", "submitted", "cli:op", {}),
		ev(2, "fb_2", "applied", "user:bob", {"forced": true, "reason": ""}),
		ev(3, "fb_2", "rolled_back", "admin:phil", {"reason": ""}),
	]
	r := cc.report with input as {"events": bad, "chain": {"ok": false}}
	count(r.violations) >= 5
	every v in r.violations {
		startswith(v.rule, "CC-")
		count(v.controls) > 0
		every c in v.controls {
			regex.match(`^[A-Z]{2}-[0-9]+$`, c)
		}
	}
}

test_approver_equal_to_submitter_is_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "cli:alger", {}),
		ev(2, "fb_1", "approved", "cli:alger", {}),
	]
	r := run(log)
	"CC-SOD" in rules_hit(r)
}

test_same_person_in_a_different_role_is_still_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "admin:alice", {}),
	]
	"CC-SOD" in rules_hit(run(log))
}

test_a_different_approver_is_fine if {
	not "CC-SOD" in rules_hit(run(good_log))
}

test_single_operator_mode_reports_an_exception_not_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "cli:alger", {}),
		ev(2, "fb_1", "approved", "cli:alger", {}),
	]
	r := cc.report with input as {"events": log, "chain": ok_chain, "options": {"single_operator": true}}
	not "CC-SOD" in rules_hit(r)
	some e in r.exceptions
	e.rule == "CC-SOD"
	e.controls == ["AC-5"]
	r.single_operator == true
}

test_applying_without_any_approval_is_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "applied", "admin:phil", {"gate": {"passed": true}}),
	]
	"CC-APPLY-APPROVED" in rules_hit(run(log))
}

test_approval_after_the_apply_does_not_count if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "applied", "admin:phil", {"gate": {"passed": true}}),
		ev(3, "fb_1", "approved", "admin:phil", {}),
	]
	"CC-APPLY-APPROVED" in rules_hit(run(log))
}

test_apply_without_a_gate_result_is_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "admin:phil", {}),
		ev(3, "fb_1", "applied", "admin:phil", {}),
	]
	"CC-APPLY-GATE" in rules_hit(run(log))
}

test_a_failed_gate_without_a_forced_override_is_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "admin:phil", {}),
		ev(3, "fb_1", "applied", "admin:phil", {"gate": {"passed": false}}),
	]
	"CC-APPLY-GATE" in rules_hit(run(log))
}

test_a_forced_override_with_a_reason_is_accepted if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "admin:phil", {}),
		ev(3, "fb_1", "applied", "admin:phil", {"gate": {"passed": false}, "forced": true, "reason": "accepted risk"}),
	]
	hits := rules_hit(run(log))
	not "CC-APPLY-GATE" in hits
	not "CC-FORCED-REASON" in hits
}

test_a_forced_override_without_a_reason_is_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "admin:phil", {}),
		ev(3, "fb_1", "applied", "admin:phil", {"gate": {"passed": false}, "forced": true, "reason": null}),
	]
	"CC-FORCED-REASON" in rules_hit(run(log))
}

test_a_rollback_needs_a_reason if {
	base := array.concat(good_log, [ev(4, "fb_1", "rolled_back", "admin:phil", {})])
	"CC-ROLLBACK-REASON" in rules_hit(run(base))
	ok := array.concat(good_log, [ev(4, "fb_1", "rolled_back", "admin:phil", {"reason": "wrong label"})])
	not "CC-ROLLBACK-REASON" in rules_hit(run(ok))
}

test_review_actions_by_a_plain_user_are_a_violation if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "user:bob", {}),
	]
	"CC-ROLE" in rules_hit(run(log))
}

test_cli_operators_and_admins_may_review if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "cli:phil", {}),
	]
	not "CC-ROLE" in rules_hit(run(log))
}

test_events_without_an_actor_or_timestamp_are_violations if {
	no_actor := [object.union(ev(1, "fb_1", "submitted", "x", {}), {"actor": ""})]
	"CC-EVENT-CONTENT" in rules_hit(run(no_actor))
	no_ts := [object.remove(ev(1, "fb_1", "submitted", "user:a", {}), ["ts"])]
	"CC-EVENT-CONTENT" in rules_hit(run(no_ts))
}

test_a_broken_chain_is_a_violation if {
	r := cc.report with input as {"events": good_log, "chain": {"ok": false}}
	"CC-CHAIN" in rules_hit(r)
}

test_a_missing_chain_result_is_treated_as_unverified if {
	r := cc.report with input as {"events": good_log}
	"CC-CHAIN" in rules_hit(r)
}

test_non_feedback_events_are_ignored_by_the_workflow_rules if {
	log := [{"seq": 1, "ts": "t", "kind": "audit", "entity_id": "tok_1", "event": "token_created", "actor": "cli:phil", "payload": {}}]
	r := run(log)
	count(r.violations) == 0
}

test_actors_without_a_role_prefix_are_compared_by_their_whole_name if {
	log := [
		ev(1, "fb_1", "submitted", "alice", {}),
		ev(2, "fb_1", "approved", "alice", {}),
	]
	"CC-SOD" in rules_hit(run(log))
	other := [
		ev(1, "fb_1", "submitted", "alice", {}),
		ev(2, "fb_1", "approved", "bob", {}),
	]
	not "CC-SOD" in rules_hit(run(other))
}

test_the_legacy_plain_cli_actor_counts_as_the_cli_operator if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "cli", {}),
		ev(3, "fb_1", "applied", "cli", {"gate": {"passed": true}}),
	]
	hits := rules_hit(run(log))
	not "CC-ROLE" in hits
	not "CC-APPLY-APPROVED" in hits
}

test_an_unknown_plain_actor_is_still_not_privileged if {
	log := [
		ev(1, "fb_1", "submitted", "user:alice", {}),
		ev(2, "fb_1", "approved", "root", {}),
	]
	"CC-ROLE" in rules_hit(run(log))
}
