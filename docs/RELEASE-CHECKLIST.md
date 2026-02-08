# Release 检查清单 (RELEASE-CHECKLIST)

## 发布流程

### 1. 创建 Tag

```bash
# 确保在 main/master 分支
git checkout main
git pull origin main

# 创建 tag
git tag -a v2.0.2 -m "Release v2.0.2 - <description>"

# 推送 tag
git push origin v2.0.2
```

### 2. 验证 CI

```bash
# 等待 CI 完成
gh run watch <tag-ci-run-id> --compact --exit-status

# 确认所有 jobs 通过
gh run view <tag-ci-run-id>
```

### 3. 创建 Release

```bash
# 使用 GitHub CLI 创建 release
gh release create v2.0.2 \
  --title "Release v2.0.2" \
  --notes "Release notes here"

# 或手动在 GitHub 上创建
```

### 4. 分支保护

```bash
# 设置分支保护规则（需管理员权限）
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --input - <<< '{
    "required_status_checks": {
      "strict": true,
      "contexts": ["lint", "test", "e2e", "verify"]
    },
    "enforce_admins": true,
    "required_pull_request_reviews": {
      "required_approving_review_count": 1
    }
  }'
```

### 5. Required Checks

确保以下 checks 为 required:
- [ ] lint
- [ ] test
- [ ] e2e
- [ ] verify
- [ ] gitleaks (Security)

### 6. Dependabot

```bash
# 检查 dependabot 配置
cat .github/dependabot.yml

# 确保启用:
# - pip dependencies
# - GitHub Actions
```

### 7. Security 扫描

```bash
# 检查 security 扫描结果
gh run list --workflow=Security --limit 5

# 确认 gitleaks 通过
gh run view <security-run-id> --log | grep -i "gitleaks\|leaks"
```

## 验收命令

```bash
# 验证 tag 存在
git ls-remote --tags origin | grep v2.0.2

# 验证 release 存在
gh release view v2.0.2

# 验证 CI 通过
gh run list --limit 5 | grep v2.0.2
```

## 回滚方案

```bash
# 删除 tag
git push origin --delete v2.0.2
git tag -d v2.0.2

# 删除 release
gh release delete v2.0.2 --yes
```

## 检查清单

- [ ] Tag 已创建并推送
- [ ] CI 全部通过
- [ ] Release 已创建
- [ ] 分支保护已设置
- [ ] Required checks 已配置
- [ ] Dependabot 已启用
- [ ] Security 扫描通过
