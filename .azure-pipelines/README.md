# Azure Pipelines Configuration

This directory contains Azure DevOps pipeline definitions for hexDAG.

## Pipelines

### 1. semantic-version.yml - Automated Semantic Versioning

**Trigger**: Commits to `main` branch

**What it does**:
1. Analyzes commit messages using Conventional Commits
2. Automatically bumps version (patch/minor/major)
3. Updates `pyproject.toml` with new version
4. Generates/updates CHANGELOG.md
5. Builds the package
6. Publishes to Azure Artifacts
7. Creates git tag (e.g., `v0.2.0`)
8. Pushes tag and version bump commit back to repo

**Commit message format** (Conventional Commits):
```bash
# Patch bump (0.1.0 → 0.1.1)
git commit -m "fix: resolve database connection issue"

# Minor bump (0.1.0 → 0.2.0)
git commit -m "feat: add new ETL connector for Parquet files"

# Major bump (0.1.0 → 1.0.0)
git commit -m "feat!: redesign API with breaking changes"
# or
git commit -m "feat: redesign API

BREAKING CHANGE: API endpoints have changed"
```

**Setup**:
1. Go to Azure DevOps → Pipelines → New pipeline
2. Select your repository
3. Choose "Existing Azure Pipelines YAML file"
4. Select `.azure-pipelines/semantic-version.yml`
5. Grant pipeline permission to push to repository:
   - Project Settings → Repositories → hexDAG
   - Security → Build Service
   - Allow "Contribute" and "Create tag"
6. Grant pipeline permission to Azure Artifacts feed:
   - Artifacts → Feed settings → Permissions
   - Add "hexDAG Build Service"
   - Role: "Contributor"

## Conventional Commits Cheat Sheet

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types:
- **feat**: New feature (triggers minor version bump)
- **fix**: Bug fix (triggers patch version bump)
- **docs**: Documentation changes
- **style**: Code style changes (formatting, no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding/updating tests
- **build**: Build system changes
- **ci**: CI/CD changes
- **chore**: Other changes (dependencies, etc.)

### Breaking Changes:
Add `!` after type or `BREAKING CHANGE:` in footer (triggers major version bump)

```bash
feat!: remove deprecated API endpoints

BREAKING CHANGE: The /v1/users endpoint has been removed
```

## Local Development

### Test version bumping locally:

```bash
# See what version bump would occur
cz bump --dry-run

# Bump version (creates commit and tag)
cz bump

# Manually specify bump type
cz bump --increment PATCH   # 0.1.0 → 0.1.1
cz bump --increment MINOR   # 0.1.0 → 0.2.0
cz bump --increment MAJOR   # 0.1.0 → 1.0.0
```

### Build and publish manually:

```bash
# Build
python -m build

# Publish to Azure Artifacts
export TWINE_USERNAME=user
export TWINE_PASSWORD=your_pat_token
twine upload -r azure-hexdag dist/*
```

## Pipeline Permissions

For semantic-version.yml to work, the build service needs permissions:

1. **Repository Permissions**:
   - Go to: Project Settings → Repositories → hexDAG → Security
   - Find: "hexDAG Build Service (omniviser)"
   - Grant:
     - ✅ Contribute
     - ✅ Create tag
     - ✅ Read

2. **Pipeline Variable** (optional):
   - Use `$(System.AccessToken)` for authentication
   - Already configured in the pipeline

## Troubleshooting

### "Permission denied" when pushing tags
- Check build service has "Contribute" and "Create tag" permissions
- Verify "Limit job authorization scope" is disabled in pipeline settings

### Version not bumping
- Check commit messages follow Conventional Commits format
- Run `cz bump --dry-run` locally to see what would happen
- Ensure no uncommitted changes in working directory

### Package not published
- Check Azure Artifacts feed exists and is named correctly
- Verify TwineAuthenticate task is configured with correct feed name
- Check build service has Contributor role on the feed

## References

- [Conventional Commits](https://www.conventionalcommits.org/)
- [Commitizen](https://commitizen-tools.github.io/commitizen/)
- [Azure Pipelines YAML Schema](https://docs.microsoft.com/en-us/azure/devops/pipelines/yaml-schema)
