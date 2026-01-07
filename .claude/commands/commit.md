---
description: Stage all changes, commit with auto-generated message, and push to GitHub
---

# Git Commit & Push

Stage all changes, generate a descriptive commit message, and push to GitHub.

## Steps

1. Run `git status` to see all untracked and modified files
2. Run `git diff` to understand what changed (both staged and unstaged)
3. Run `git log -3 --oneline` to see recent commit message style
4. Stage all relevant files with `git add` (exclude secrets, .env files, credentials)
5. Generate a commit message that:
   - Starts with a verb (Add, Fix, Update, Refactor, Remove, etc.)
   - Summarizes the nature of changes concisely (1-2 sentences)
   - Focuses on the "why" not just the "what"
   - Ends with the Claude Code signature block
6. Create the commit using HEREDOC format for the message
7. Push to origin with `git push`
8. Report the commit hash and confirmation of push

## Commit Message Format

```
<type>: <short description>

<optional longer description if needed>

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

## Safety Rules

- NEVER commit .env files, credentials, or secrets
- NEVER use --force push
- NEVER skip pre-commit hooks (--no-verify)
- If there are no changes, report "Nothing to commit"
