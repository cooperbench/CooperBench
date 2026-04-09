**Title**: Add readonly parameter to click.edit()

**Pull Request Details**

**Description**:
This change introduces a `readonly` parameter to the `click.edit()` function, allowing users to open files in a read-only mode. The implementation detects common editors like Vim and Nano to apply their specific read-only flags.

**Technical Background**:
While `click.edit()` provides a convenient way to interface with system editors, it previously lacked a mechanism to signal that a file should not be modified. Many command-line editors support specific flags to prevent accidental writes (like "view mode"), which is useful when using Click to display logs or configuration files that should remain immutable.

**Solution**:
A new `readonly` boolean parameter was added to `click.edit()` and the underlying `Editor` class. When set to `True`, the implementation modifies the shell command string based on the detected editor:
- For **Vim** or **Vi**, it appends the `-R` flag.
- For **Nano**, it appends the `-v` flag.
If the editor is not recognized, it falls back to the standard command execution to ensure compatibility.

**Files Modified**
- `src/click/_termui_impl.py`
- `src/click/termui.py`