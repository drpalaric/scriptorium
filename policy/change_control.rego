# METADATA
# title: Change control over model corrections
# description: >
#   Evaluates the exported audit log of the corrections workflow. Every rule carries the NIST SP 800-53
#   controls it supports in `custom.controls`. The control assignment is the author's judgment, not an
#   official mapping.
package scriptorium.change_control

# Pull the facts out of each event once.
feedback_events := [e | some e in input.events; e.kind == "feedback"]

entity_ids := {e.entity_id | some e in feedback_events}

events_of(id) := [e | some e in feedback_events; e.entity_id == id]

blank(x) if x == null

blank(x) if trim_space(x) == ""

name_of(actor) := actor if not contains(actor, ":")

name_of(actor) := split(actor, ":")[1] if contains(actor, ":")

payload_field(e, field) := object.get(e.payload, field, null)

# METADATA
# title: The approver must not be the submitter
# custom:
#   id: CC-SOD
#   controls: [AC-5]
#   mapping_basis: author's judgment (separation of duties)
sod_findings contains v if {
	some id in entity_ids
	evs := events_of(id)
	some s in evs
	s.event == "submitted"
	some a in evs
	a.event == "approved"
	name_of(a.actor) == name_of(s.actor)
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": id,
		"message": sprintf("%s approved a correction that %s submitted", [a.actor, s.actor]),
	}
}

# Separation-of-duties findings are violations, unless the operator states this is a single-operator
# deployment; then they are reported as accepted exceptions, never silently dropped.
violations contains v if {
	some v in sod_findings
	not input.options.single_operator
}

exceptions contains v if {
	some v in sod_findings
	input.options.single_operator
}

# METADATA
# title: A correction may be applied only after it was approved
# custom:
#   id: CC-APPLY-APPROVED
#   controls: [CM-3, CM-5]
#   mapping_basis: author's judgment (change control, access restrictions for change)
violations contains v if {
	some e in feedback_events
	e.event == "applied"
	not approved_before(e)
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": e.entity_id,
		"message": "applied without an earlier approval",
	}
}

approved_before(e) if {
	some a in feedback_events
	a.entity_id == e.entity_id
	a.event == "approved"
	a.seq < e.seq
}

# METADATA
# title: An applied change needs a recorded held-out result that passed, or a forced override
# custom:
#   id: CC-APPLY-GATE
#   controls: [CM-4, SA-11]
#   mapping_basis: author's judgment (impact analysis, testing before release)
violations contains v if {
	some e in feedback_events
	e.event == "applied"
	not gate_ok(e)
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": e.entity_id,
		"message": "applied without a passing recorded gate result",
	}
}

gate_ok(e) if payload_field(e, "gate").passed == true

gate_ok(e) if payload_field(e, "forced") == true

# METADATA
# title: A forced override needs a stated reason
# custom:
#   id: CC-FORCED-REASON
#   controls: [CM-3]
#   mapping_basis: author's judgment (documented change decisions)
violations contains v if {
	some e in feedback_events
	e.event == "applied"
	payload_field(e, "forced") == true
	blank(payload_field(e, "reason"))
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": e.entity_id,
		"message": "forced override without a reason",
	}
}

# METADATA
# title: A rollback needs a stated reason
# custom:
#   id: CC-ROLLBACK-REASON
#   controls: [CM-3]
#   mapping_basis: author's judgment (documented change decisions)
violations contains v if {
	some e in feedback_events
	e.event == "rolled_back"
	blank(payload_field(e, "reason"))
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": e.entity_id,
		"message": "rolled back without a reason",
	}
}

# METADATA
# title: Review and change actions must come from an administrator or the CLI operator
# custom:
#   id: CC-ROLE
#   controls: [AC-6, CM-5]
#   mapping_basis: author's judgment (least privilege, access restrictions for change)
violations contains v if {
	some e in feedback_events
	e.event in {"approved", "rejected", "applied", "rolled_back"}
	not privileged(e.actor)
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": e.entity_id,
		"message": sprintf("%s performed %s without an administrator role", [e.actor, e.event]),
	}
}

privileged(actor) if startswith(actor, "admin:")

privileged(actor) if startswith(actor, "cli:")

# Older audit events recorded the CLI operator as plain "cli" (no user name).
privileged(actor) if actor == "cli"

# METADATA
# title: Every audit event records who did it and when
# custom:
#   id: CC-EVENT-CONTENT
#   controls: [AU-3]
#   mapping_basis: author's judgment (content of audit records)
violations contains v if {
	some e in input.events
	missing_content(e)
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": sprintf("seq %d", [e.seq]),
		"message": "event is missing its actor or timestamp",
	}
}

missing_content(e) if blank(object.get(e, "actor", null))

missing_content(e) if blank(object.get(e, "ts", null))

# METADATA
# title: The audit log's hash chain must verify before it is relied on
# custom:
#   id: CC-CHAIN
#   controls: [SI-7, AU-9]
#   mapping_basis: author's judgment (integrity, protection of audit information)
violations contains v if {
	not input.chain.ok == true
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"entity": "audit log",
		"message": "the audit log hash chain does not verify",
	}
}

report := {
	"violations": violations,
	"exceptions": exceptions,
	"events_checked": count(input.events),
	"single_operator": object.get(object.get(input, "options", {}), "single_operator", false),
}
