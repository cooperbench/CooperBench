# Title: feat(mux): add request-based route selection with dynamic routing

**Description:**

This PR adds dynamic routing capabilities to chi, allowing applications to
select different routing handlers based on request properties beyond just
the URL path.

This allows for:

- Version-based routing (e.g., via Accept header)
- User role-based routing (selecting different handlers based on auth context)
- Feature flag-based routing (enabling/disabling endpoints dynamically)

The implementation adds a new request context evaluation step during route matching
that allows applications to customize route selection based on any criteria.

## Technical Background

### Issue Context:

Modern APIs often need to route requests dynamically based on factors like API versioning, feature flags, or user permissions. Traditional static routing based solely on URL patterns is insufficient for these use cases.

This feature enables request-based dynamic routing by adding a context evaluation phase to the routing process. This allows applications to examine the request context (headers, auth info, etc.) and select the appropriate handler accordingly.

## Public API

Callers depend on these exact names.

```go
// A selector decides which handler serves a pattern, given the request.
type RouteSelector interface {
    SelectRoute(r *http.Request, pattern string, defaultHandler http.Handler) http.Handler
}

// Attaches a selector to a request, for use from middleware.
func WithRouteSelector(r *http.Request, selector RouteSelector) *http.Request

// Chooses by API version, read from the request.
func NewVersionSelector(defaultVersion string) *VersionSelector
func (vs *VersionSelector) AddHandler(pattern, version string, handler http.Handler)

// Chooses by caller role, extracted by the supplied function.
func NewRoleBasedSelector(roleExtractor func(r *http.Request) string, defaultRole string) *RoleBasedSelector
func (rs *RoleBasedSelector) AddHandler(pattern, role string, handler http.Handler)
```

When no selector is attached, or the selector returns nothing for the pattern, the route's own
handler serves the request as before.

## Files Modified

```
- mux.go
- context.go
- dynamic_route.go
- dynamic_route_test.go
```
