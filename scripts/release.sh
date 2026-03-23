#!/bin/bash
# Release script — bumps version, updates changelog, builds, and publishes.
# Usage: ./scripts/release.sh 0.2.0

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/release.sh <version>"
    echo "Example: ./scripts/release.sh 0.2.0"
    exit 1
fi

echo "=== Releasing ZeroLLM v${VERSION} ==="

# 1. Update version in pyproject.toml and __init__.py
sed -i '' "s/^version = .*/version = \"${VERSION}\"/" pyproject.toml
sed -i '' "s/__version__ = .*/__version__ = \"${VERSION}\"/" zerollm/__init__.py
echo "✓ Version bumped to ${VERSION}"

# 2. Auto-generate changelog from git commits
if command -v git-cliff &> /dev/null; then
    git-cliff --tag "v${VERSION}" --output CHANGELOG.md
    echo "✓ Changelog updated"
else
    echo "! git-cliff not installed, skipping auto-changelog"
    echo "  Install: cargo install git-cliff  OR  uv tool install git-cliff"
fi

# 3. Commit version bump
git add pyproject.toml zerollm/__init__.py CHANGELOG.md
git commit -m "release: v${VERSION}"
git tag -a "v${VERSION}" -m "ZeroLLM v${VERSION}"
echo "✓ Tagged v${VERSION}"

# 4. Build
rm -rf dist/
uv build
echo "✓ Built"

# 5. Confirm before publishing
echo ""
echo "Ready to publish v${VERSION} to PyPI."
read -p "Publish? (y/N) " confirm
if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    uv publish --username __token__ --password "$(grep password ~/.pypirc | awk '{print $3}')"
    git push && git push --tags
    echo "✓ Published to PyPI and pushed to GitHub"
else
    echo "Aborted. Run manually:"
    echo "  uv publish"
    echo "  git push && git push --tags"
fi
