# Tag-Based PyPI Deployment Guide

This repository uses **tag-based deployment** to automatically publish packages to PyPI. This ensures that only intentional, versioned releases are published.

## How It Works

When you create and push a version tag (e.g., `v0.1.0`), GitHub Actions will:

1. ✅ Verify the tag version matches `pyproject.toml`
2. 🔨 Build the package
3. ✔️ Run quality checks with twine
4. 📦 Publish to PyPI
5. 🎉 Create a GitHub Release with the package files

## Publishing a New Version

### Step 1: Update Version Number

Update the version in `pyproject.toml`:

```toml
[project]
name = "gptmed"
version = "0.1.0"  # Change this to your new version
```

Also update `gptmed/__init__.py`:

```python
__version__ = "0.1.0"  # Keep in sync with pyproject.toml
```

### Step 2: Update CHANGELOG.md

Add a new section for your version:

```markdown
## [0.1.0] - 2026-01-11

### Added
- New feature X
- New feature Y

### Fixed
- Bug fix Z

### Changed
- Updated documentation
```

### Step 3: Commit Your Changes

```bash
git add pyproject.toml gptmed/__init__.py CHANGELOG.md
git commit -m "Bump version to 0.1.0"
git push origin main
```

### Step 4: Create and Push a Tag

```bash
# Create a tag matching the version (with 'v' prefix)
git tag -a v0.1.0 -m "Release version 0.1.0"

# Push the tag to GitHub
git push origin v0.1.0
```

That's it! The GitHub Action will automatically:
- Build the package
- Publish to PyPI
- Create a GitHub Release

## Tag Naming Convention

Tags **must** follow this format:

- ✅ `v0.1.0` - Correct
- ✅ `v1.2.3` - Correct
- ✅ `v2.0.0-beta.1` - Correct (pre-release)
- ❌ `0.1.0` - Missing 'v' prefix
- ❌ `version-0.1.0` - Wrong format
- ❌ `release-1.0` - Wrong format

## Version Validation

The workflow includes a **version check** that ensures:

```
pyproject.toml version = "0.1.0"
Git tag = "v0.1.0"
```

If they don't match, the deployment will fail. This prevents accidental mismatched releases.

## Testing Before PyPI Release

### Option 1: Test PyPI (Recommended)

Before creating a production tag, test on Test PyPI:

1. Update version to include a pre-release suffix:
   ```toml
   version = "0.1.0rc1"  # Release candidate
   ```

2. Create a pre-release tag:
   ```bash
   git tag -a v0.1.0rc1 -m "Release candidate 0.1.0rc1"
   git push origin v0.1.0rc1
   ```

3. Manually trigger the workflow with Test PyPI enabled

### Option 2: Manual Build

Test locally before tagging:

```bash
# Build the package
python -m build

# Check with twine
twine check dist/*

# Test installation locally
pip install dist/gptmed-0.1.0-py3-none-any.whl
```

## Deleting a Tag (If Needed)

If you made a mistake:

```bash
# Delete local tag
git tag -d v0.1.0

# Delete remote tag
git push origin :refs/tags/v0.1.0
```

Then fix the issue and create a new tag.

## Semantic Versioning

Follow [Semantic Versioning](https://semver.org/):

- `v1.0.0` - **MAJOR**: Breaking changes
- `v0.2.0` - **MINOR**: New features (backward compatible)
- `v0.1.1` - **PATCH**: Bug fixes

Examples:
- `v0.1.0` → `v0.1.1` - Bug fix
- `v0.1.0` → `v0.2.0` - New feature
- `v0.9.0` → `v1.0.0` - Stable release with breaking changes

## Pre-release Versions

For alpha, beta, or release candidates:

- `v0.1.0-alpha.1` - Alpha release
- `v0.1.0-beta.2` - Beta release  
- `v0.1.0-rc.3` - Release candidate

## Quick Reference

```bash
# Complete release workflow
vim pyproject.toml           # Update version
vim gptmed/__init__.py       # Update __version__
vim CHANGELOG.md             # Add changelog entry
git add pyproject.toml gptmed/__init__.py CHANGELOG.md
git commit -m "Bump version to 0.1.0"
git push origin main
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0       # This triggers PyPI deployment!
```

## Troubleshooting

### Workflow Fails: Version Mismatch

**Error**: `Version mismatch! pyproject.toml has version 0.1.0 but tag is v0.2.0`

**Fix**: Make sure the version in `pyproject.toml` matches your tag (without the 'v'):
- Tag: `v0.1.0`
- pyproject.toml: `version = "0.1.0"`

### Workflow Fails: PyPI Upload Error

**Error**: `File already exists`

**Cause**: You're trying to upload a version that already exists on PyPI

**Fix**: Increment the version number and create a new tag

### Tag Doesn't Trigger Workflow

**Cause**: Tag format doesn't match `v*.*.*` pattern

**Fix**: Tag must start with 'v' and follow semver (e.g., `v0.1.0`)

## GitHub Secrets Required

Make sure these secrets are configured in your repository settings:

- `PYPI_API_TOKEN` - PyPI API token (required)
- `TEST_PYPI_API_TOKEN` - Test PyPI token (optional, for testing)

To add secrets:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add your PyPI API tokens

## Benefits of Tag-Based Deployment

✅ **Intentional releases** - Only publish when you create a tag  
✅ **Version control** - Git tags track every release  
✅ **Rollback capability** - Easy to see what was released when  
✅ **No accidental publishes** - Won't publish on every commit  
✅ **Clean release history** - GitHub Releases for each version  
✅ **Automated** - Just push a tag, everything else is automatic  

---

**Happy releasing! 🚀**
