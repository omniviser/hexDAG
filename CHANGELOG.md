## Version Release: 0.1.0
Merged PR 2166: feat: add cz

feat: add cz

## Version Release: 0.1.1
Merged PR 2168: fix: add uv

fix: add uv

## Version Release: 0.1.2
Merged PR 2170: test: JEAN PLS MERGE ME TO TEST THE CI PIPELINE

If you merge PR there will be bump based on the prefix in PR title
```java
bump_map = {
  ".*!" = "MAJOR",
  "ci" = "PATCH",
  "docs" = "PATCH",
  "experiment" = "PATCH",
  "feat" = "MINOR",
  "fix" = "PATCH",
  "refactor" = "PATCH",
  "test" = "PATCH"
}
```
And the `CHANGELOG.md` file will be updated with the PR description and version that was bumped

In ci-pipeline.yaml there is a place to put logic for jobs triggered once per version update of the project

Related work items: #2082
