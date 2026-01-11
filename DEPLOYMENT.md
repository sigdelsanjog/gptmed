# Deployment Guide for GPTMed Package

This guide explains how to deploy new versions of the gptmed package to PyPI using the automated GitHub Actions workflow.

## Prerequisites

1. **PyPI Account**: You need a PyPI account at https://pypi.org
2. **PyPI API Token**: Generate an API token from your PyPI account settings
3. **GitHub Repository**: The code must be pushed to GitHub
4. **GitHub Secret**: Store your PyPI API token as a GitHub secret

## Setup (One-time)

### 1. Generate PyPI API Token

1. Log in to https://pypi.org
2. Go to Account Settings → API tokens
3. Click "Add API token"
4. Name it (e.g., "gptmed-github-actions")
5. Scope: "Entire account" or specific to "gptmed" project
6. Copy the token (starts with `pypi-`)

### 2. Add GitHub Secret

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Paste your PyPI API token
6. Click "Add secret"

### 3. (Optional) Test PyPI Setup

For testing before deploying to production PyPI:

1. Create account at https://test.pypi.org
2. Generate API token
3. Add as GitHub secret: `TEST_PYPI_API_TOKEN`

## Deployment Process

### Step 1: Update Version

Update the version in **two** files:

**File 1: `pyproject.toml`**

```toml
[project]
name = "gptmed"
version = "0.3.0"  # ← Update this
```

**File 2: `gptmed/__init__.py`**

```python
__version__ = "0.3.0"  # ← Update this
```

**Important**: Both versions must match!

### Step 2: Commit Changes

```bash
cd gptmed
git add pyproject.toml gptmed/__init__.py
git commit -m "Bump version to 0.3.0"
git push origin main
```

### Step 3: Create and Push Tag

```bash
# Create annotated tag (recommended)
git tag -a v0.3.0 -m "Release version 0.3.0 with high-level API"

# Push the tag to GitHub
git push origin v0.3.0
```

**Tag Format**: Must be `v*.*.*` (e.g., `v0.3.0`, `v1.2.3`)

### Step 4: Automated Deployment

Once you push the tag, GitHub Actions will automatically:

1. ✅ Check out the code
2. ✅ Verify version in `pyproject.toml` matches the tag
3. ✅ Build the package (wheel and source distribution)
4. ✅ Check package with twine
5. ✅ Upload to PyPI
6. ✅ Create a GitHub Release with the built artifacts

### Step 5: Monitor Deployment

1. Go to your GitHub repository
2. Click "Actions" tab
3. Watch the "Publish to PyPI" workflow
4. Check for any errors

### Step 6: Verify on PyPI

1. Visit https://pypi.org/project/gptmed/
2. Verify the new version is live
3. Check that the API functions are documented

## Manual Deployment (Alternative)

If you need to deploy manually without GitHub Actions:

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Install build tools (use older setuptools to avoid metadata issues)
pip install "setuptools<70" wheel build twine

# Build the package
python -m build

# Check the package
twine check dist/*

# Upload to PyPI
twine upload dist/*
```

You'll be prompted for your PyPI username and password/token.

## Testing Before Production

To test on Test PyPI first:

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ gptmed

# Test the installation
python -c "import gptmed; print(gptmed.__version__); print(dir(gptmed))"
```

## Post-Deployment Verification

After deployment, verify the package works:

```bash
# Create a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install gptmed

# Test the API
python -c "
import gptmed
print(f'Version: {gptmed.__version__}')
print(f'API functions: {gptmed.__all__}')

# Test that API functions exist
assert hasattr(gptmed, 'create_config')
assert hasattr(gptmed, 'train_from_config')
assert hasattr(gptmed, 'generate')
print('✅ All API functions available!')
"
```

## Versioning Guidelines

Follow Semantic Versioning (SemVer):

- **Major** (x.0.0): Breaking changes
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes, backward compatible

Examples:

- `v0.3.0` → Added high-level API (new feature)
- `v0.3.1` → Fixed bug in generate() (bug fix)
- `v1.0.0` → Stable release, production ready (major milestone)

## Troubleshooting

### Version Mismatch Error

**Error**: "Version mismatch! pyproject.toml has version X but tag is vY"

**Solution**: Make sure both files have the same version:

- `pyproject.toml`: `version = "0.3.0"`
- `gptmed/__init__.py`: `__version__ = "0.3.0"`
- Git tag: `v0.3.0`

### License Metadata Error

**Error**: "Invalid distribution metadata: unrecognized or malformed field 'license-expression'"

**Solution**: The workflow uses `setuptools<70` to avoid this. If building locally:

```bash
pip install "setuptools<70" wheel
python -m build
```

### PYPI_API_TOKEN Not Found

**Error**: "Error: Input required and not supplied: password"

**Solution**:

1. Verify the GitHub secret is named exactly `PYPI_API_TOKEN`
2. Check it's set at repository level, not organization level
3. Re-generate the PyPI token if needed

### Package Already Exists

**Error**: "File already exists"

**Solution**: You cannot re-upload the same version. Increment the version number.

## Quick Reference

```bash
# Complete release process
git add pyproject.toml gptmed/__init__.py
git commit -m "Bump version to X.Y.Z"
git push origin main
git tag -a vX.Y.Z -m "Release version X.Y.Z"
git push origin vX.Y.Z

# Watch the deployment
# Go to: https://github.com/YOUR_USERNAME/gptmed/actions
```

## Files Modified for Deployment

- `pyproject.toml` - Package metadata and version
- `gptmed/__init__.py` - Package version constant
- `.github/workflows/publish-to-pypi.yml` - Automated deployment workflow

## Support

If you encounter issues:

1. Check the GitHub Actions logs
2. Verify all secrets are set correctly
3. Test build locally first
4. Consult PyPI documentation: https://packaging.python.org/
