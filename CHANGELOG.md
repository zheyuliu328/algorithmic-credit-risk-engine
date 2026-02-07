# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.1] - 2026-02-08

### Added
- GitHub Actions CI with lint, test, verify
- Gitleaks security scanning in CI
- E2E tests for run-real path
- Pre-commit hooks configuration

### Changed
- Version bump from 2.0.0 to 2.0.1 (2.0.0 was never pushed to remote)
- Fixed gitleaks configuration to avoid false positives on config examples

## [2.0.0] - 2026-02-08

### Added
- src layout migration
- Makefile with standard commands
- Unified verify script
- Security CI with gitleaks and bandit
- Pre-commit hooks
- Run-real path for CSV scoring
- LICENSE (MIT)
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- SECURITY.md

### Changed
- Removed regulatory compliance claims from pyproject.toml
- Updated README with accurate project description

### Security
- Added gitleaks configuration
- Added secrets management
- Added input validation
