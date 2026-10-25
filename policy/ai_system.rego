# METADATA
# title: AI system inventory and governance records
# description: >
#   Evaluates records that describe AI systems. Every rule carries the NIST SP 800-53 controls it supports
#   in `custom.controls`. The control assignment is the author's judgment, not an official mapping.
package scriptorium.ai_system

required_fields := ["owner", "purpose", "risk_tier", "incident_contact"]

elevated := {"medium", "high"}

blank(x) if x == null

blank(x) if trim_space(x) == ""

field(s, name) := object.get(s, name, null)

# METADATA
# title: Every system has an owner, purpose, risk tier and incident contact
# custom:
#   id: AI-INVENTORY
#   controls: [CM-8, PM-5]
#   mapping_basis: author's judgment (system component inventory, system inventory)
findings contains v if {
	some s in input.systems
	some f in required_fields
	blank(field(s, f))
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": sprintf("missing %s", [f]),
	}
}

# METADATA
# title: High-risk systems need human oversight
# custom:
#   id: AI-OVERSIGHT
#   controls: [RA-3, PM-9]
#   mapping_basis: author's judgment (risk assessment, risk management strategy)
findings contains v if {
	some s in input.systems
	field(s, "risk_tier") == "high"
	not field(s, "human_oversight") == true
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": "high-risk system without human oversight",
	}
}

# METADATA
# title: Medium and high-risk systems need verified evaluation evidence (a report whose sha256 matches the file)
# custom:
#   id: AI-EVIDENCE
#   controls: [CA-2, SA-11]
#   mapping_basis: author's judgment (control assessments, developer testing)
findings contains v if {
	some s in input.systems
	field(s, "risk_tier") in elevated
	count(valid_evidence(s)) == 0
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": "no verified evaluation evidence (a report whose sha256 matches the file)",
	}
}

valid_evidence(s) := [x |
	some x in object.get(s, "evaluation_evidence", [])
	regex.match(`^[0-9a-f]{64}$`, object.get(x, "sha256", ""))
	object.get(x, "verified", false) == true
]

# METADATA
# title: An external model provider must be approved
# custom:
#   id: AI-EXTERNAL-APPROVED
#   controls: [SA-9]
#   mapping_basis: author's judgment (external system services)
findings contains v if {
	some s in input.systems
	field(s, "model_provider") == "external"
	not field(s, "external_api_approved") == true
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": "external model provider without a recorded approval",
	}
}

# METADATA
# title: Restricted data must not go to an external provider
# custom:
#   id: AI-RESTRICTED-EXTERNAL
#   controls: [AC-4, SC-7]
#   mapping_basis: author's judgment (information flow enforcement, boundary protection)
findings contains v if {
	some s in input.systems
	field(s, "model_provider") == "external"
	field(s, "data_classification") == "restricted"
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": "restricted data processed by an external provider",
	}
}

# METADATA
# title: Personal data requires a confidential or restricted classification
# custom:
#   id: AI-PERSONAL-DATA
#   controls: [RA-2]
#   mapping_basis: author's judgment (security categorization)
findings contains v if {
	some s in input.systems
	field(s, "personal_data") == true
	not field(s, "data_classification") in {"confidential", "restricted"}
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": "personal data below a confidential classification",
	}
}

# METADATA
# title: Medium and high-risk systems must log their activity
# custom:
#   id: AI-LOGGING
#   controls: [AU-2, AU-12]
#   mapping_basis: author's judgment (event logging, audit record generation)
findings contains v if {
	some s in input.systems
	field(s, "risk_tier") in elevated
	not field(s, "logging_enabled") == true
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": "logging is not enabled",
	}
}

# METADATA
# title: Reviews must be current
# custom:
#   id: AI-REVIEW
#   controls: [CA-7, RA-3]
#   mapping_basis: author's judgment (continuous monitoring, risk assessment updates)
findings contains v if {
	some s in input.systems
	days_since_review(s) > object.get(s, "review_interval_days", 365)
	v := {
		"rule": rego.metadata.rule().custom.id,
		"controls": rego.metadata.rule().custom.controls,
		"system": object.get(s, "id", "unknown"),
		"message": sprintf("last review %d days ago", [days_since_review(s)]),
	}
}

days_since_review(s) := floor((time.parse_ns("2006-01-02", input.today) - time.parse_ns("2006-01-02", s.last_review)) / ((24 * 3600) * 1000000000))

report := {"findings": findings, "systems_checked": count(input.systems)}
