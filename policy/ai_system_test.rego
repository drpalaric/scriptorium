package scriptorium.ai_system_test

import data.scriptorium.ai_system as ai

sha := "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

good := {
	"id": "sys-1",
	"owner": "Phil",
	"purpose": "classify requirements",
	"risk_tier": "medium",
	"incident_contact": "phil@example.invalid",
	"model_provider": "internal",
	"data_classification": "internal",
	"personal_data": false,
	"human_oversight": true,
	"logging_enabled": true,
	"evaluation_evidence": [{"report": "r.json", "sha256": sha, "verified": true}],
	"last_review": "2026-10-01",
	"review_interval_days": 365,
}

run(sys) := r if {
	r := ai.report with input as {"systems": [sys], "today": "2026-10-25"}
}

hits(r) := {f.rule | some f in r.findings}

test_a_complete_system_has_no_findings if {
	r := run(good)
	count(r.findings) == 0
	r.systems_checked == 1
}

test_every_finding_carries_a_rule_id_and_nist_controls if {
	bad := {"id": "bad", "risk_tier": "high", "model_provider": "external", "data_classification": "restricted", "personal_data": true, "last_review": "2020-01-01"}
	r := run(bad)
	count(r.findings) >= 6
	every f in r.findings {
		startswith(f.rule, "AI-")
		count(f.controls) > 0
		every c in f.controls {
			regex.match(`^[A-Z]{2}-[0-9]+$`, c)
		}
	}
}

test_missing_owner_purpose_tier_or_contact_is_flagged_per_field if {
	r := run({"id": "x", "risk_tier": "low"})
	msgs := {f.message | some f in r.findings; f.rule == "AI-INVENTORY"}
	msgs == {"missing owner", "missing purpose", "missing incident_contact"}
}

test_blank_strings_count_as_missing if {
	"AI-INVENTORY" in hits(run(object.union(good, {"owner": "  "})))
}

test_high_risk_without_oversight_is_flagged if {
	s := object.union(good, {"risk_tier": "high", "human_oversight": false})
	"AI-OVERSIGHT" in hits(run(s))
	not "AI-OVERSIGHT" in hits(run(object.union(good, {"risk_tier": "high"})))
	not "AI-OVERSIGHT" in hits(run(object.union(good, {"human_oversight": false})))
}

test_medium_or_high_risk_needs_hashed_evidence if {
	s := object.remove(good, ["evaluation_evidence"])
	"AI-EVIDENCE" in hits(run(s))
	short := object.union(good, {"evaluation_evidence": [{"report": "r", "sha256": "abc", "verified": true}]})
	"AI-EVIDENCE" in hits(run(short))
	unverified := object.union(good, {"evaluation_evidence": [{"report": "r", "sha256": sha, "verified": false}]})
	"AI-EVIDENCE" in hits(run(unverified))
	never_checked := object.union(good, {"evaluation_evidence": [{"report": "r", "sha256": sha}]})
	"AI-EVIDENCE" in hits(run(never_checked))
	not "AI-EVIDENCE" in hits(run(object.union(s, {"risk_tier": "low"})))
	not "AI-EVIDENCE" in hits(run(good))
}

test_external_provider_needs_a_recorded_approval if {
	ext := object.union(good, {"model_provider": "external"})
	"AI-EXTERNAL-APPROVED" in hits(run(ext))
	not "AI-EXTERNAL-APPROVED" in hits(run(object.union(ext, {"external_api_approved": true})))
	not "AI-EXTERNAL-APPROVED" in hits(run(good))
}

test_restricted_data_may_not_go_to_an_external_provider if {
	s := object.union(good, {"model_provider": "external", "external_api_approved": true, "data_classification": "restricted"})
	"AI-RESTRICTED-EXTERNAL" in hits(run(s))
	not "AI-RESTRICTED-EXTERNAL" in hits(run(object.union(s, {"model_provider": "internal"})))
}

test_personal_data_needs_a_confidential_classification if {
	"AI-PERSONAL-DATA" in hits(run(object.union(good, {"personal_data": true})))
	not "AI-PERSONAL-DATA" in hits(run(object.union(good, {"personal_data": true, "data_classification": "confidential"})))
	not "AI-PERSONAL-DATA" in hits(run(object.union(good, {"personal_data": true, "data_classification": "restricted"})))
}

test_elevated_risk_systems_must_log if {
	"AI-LOGGING" in hits(run(object.union(good, {"logging_enabled": false})))
	not "AI-LOGGING" in hits(run(object.union(good, {"logging_enabled": false, "risk_tier": "low"})))
}

test_overdue_reviews_are_flagged_with_the_age if {
	r := run(object.union(good, {"last_review": "2025-01-01"}))
	some f in r.findings
	f.rule == "AI-REVIEW"
	contains(f.message, "days ago")
}

test_the_review_interval_is_respected if {
	not "AI-REVIEW" in hits(run(object.union(good, {"last_review": "2026-06-01", "review_interval_days": 180})))
	"AI-REVIEW" in hits(run(object.union(good, {"last_review": "2026-06-01", "review_interval_days": 100})))
}
