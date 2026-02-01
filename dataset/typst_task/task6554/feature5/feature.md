# Add count parameter to `str.first` and `str.last`

**Description:**

This feature adds a `count` parameter to `str.first()` and `str.last()` methods that returns an array of characters instead of a single character, enabling extraction of multiple characters as separate elements.

**Background:**

String processing often requires working with multiple characters individually rather than as a concatenated string. This enhancement adds array-based multi-character extraction.

**Solution:**

Modifies the existing `str.first()` and `str.last()` methods to accept an optional `count` parameter:

```rust
pub fn first(
    &self,
    /// Number of characters to return as an array. If omitted, returns a single string.
    #[named]
    count: Option<usize>,
) -> StrResult<Value>
```

**Usage Examples:**

```typst
// Original behavior preserved
"hello".first()           // → "h"
"hello".last()            // → "o"

// Array extraction
"hello".first(count: 3)   // → ("h", "e", "l")
"hello".last(count: 2)    // → ("l", "o")
"hello".first(count: 1)   // → ("h",)

// Unicode support
"🌍👋🎉".first(count: 2)   // → ("🌍", "👋")

// Array operations
#let chars = "hello".first(count: 3)
#chars.at(0)              // → "h"
#chars.len()              // → 3
```

**Technical Implementation:**

- Returns single string when count is not specified (backward compatibility)
- Returns array of strings when count is specified
- Proper error handling when count exceeds string length
- Full Unicode grapheme cluster support

**Files Modified:**
- `crates/typst/src/foundations/str.rs` - Method implementations
