# Phase 53 AI CV Generate Template Fidelity Report

Generated: `2026-06-04T21:17:17.600299+00:00`

Passed: `True`

## Cases

### valid_placeholder_fill_preserves_structure

Input template:

```html
<section id="cv" class="modern" data-kind="cv">
  <h1>{{name}}</h1>
  <p class="summary">{{summary}}</p>
  <p>Static footer</p>
</section>
```

Generated output:

```html
<section id="cv" class="modern" data-kind="cv">
  <h1></h1>
  <p class="summary">
    Backend REST API candidate with PostgreSQL delivery experience.
  </p>
  <p>Static footer</p>
</section>
```

Template diff: `{'valid': True, 'reasons': []}`
Safety checks: `{'passed': True, 'issues': []}`
Matches expectation: `True`

### invalid_provider_changes_tag_and_class

Input template:

```html
<section id="cv" class="modern">
  <h1>{{name}}</h1>
  <p>{{summary}}</p>
</section>
```

Generated output:

```html
<section id="cv" class="changed">
  <h2>Candidate</h2>
  <p>Backend REST API candidate.</p>
</section>
```

Template diff: `{'valid': False, 'reasons': ['attribute_changed_at_0_class', 'tag_changed_at_1']}`
Safety checks: `{'passed': True, 'issues': []}`
Matches expectation: `True`

### invalid_privacy_leak

Input template:

```html
<section><p>{{summary}}</p></section>
```

Generated output:

```html
<section><p>Contact me at user@example.test or +62 812 3333 4444.</p></section>
```

Template diff: `{'valid': False, 'reasons': ['unsafe_email_leak', 'unsafe_phone_leak']}`
Safety checks: `{'passed': False, 'issues': ['email_leak', 'phone_leak']}`
Matches expectation: `True`
