package xorb

import (
	"fmt"
	"regexp"
)

// Subject represents a xorb event subject
// Format: xorb.<tenant>.<domain>.<service>.<event>
type Subject struct {
	Tenant  string
	Domain  string
	Service string
	Event   string
}

var subjectRegex = regexp.MustCompile(`^xorb\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+$`)

// NewSubject creates and validates a new Subject
func NewSubject(tenant, domain, service, event string) (*Subject, error) {
	subjectStr := fmt.Sprintf("xorb.%s.%s.%s.%s", tenant, domain, service, event)
	if !subjectRegex.MatchString(subjectStr) {
		return nil, fmt.Errorf("invalid subject format: %s", subjectStr)
	}
	return &Subject{
		Tenant:  tenant,
		Domain:  domain,
		Service: service,
		Event:   event,
	}, nil
}

// String returns the string representation of the subject
func (s *Subject) String() string {
	return fmt.Sprintf("xorb.%s.%s.%s.%s", s.Tenant, s.Domain, s.Service, s.Event)
}
