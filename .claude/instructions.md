# Claude Code Instructions

You are working on an AI Deception & Persuasion Leaderboard project. This is a full-stack web application with a Python FastAPI backend (forked from existing LLM agent codebase) and a TypeScript Next.js frontend.

## Core Principles

### 1. Extreme Programming Values
- **Simplicity**: Write the simplest code that works
- **Communication**: Code should be self-documenting
- **Feedback**: Run tests constantly, commit frequently
- **Courage**: Refactor fearlessly with test coverage
- **Respect**: Leave code better than you found it

### 2. Continuous Delivery Mindset
- Every commit should be deployable
- Feature flags for incomplete features
- Linear git history (rebase, never merge)
- Small, atomic commits with clear messages

### 3. Test-Driven Development
- Write tests before implementation
- Red → Green → Refactor cycle
- Fast unit tests, selective integration tests
- E2E tests for critical user paths only

## Before Every Change

**MANDATORY Pre-flight checklist:**

```bash
# 1. Run existing tests
cd backend && pytest -v
cd frontend && bun run test

# 2. Check types
cd backend && mypy .
cd frontend && bun run type-check

# 3. Review recent commits to understand patterns
git log --oneline -10
```

**If tests fail or types don't check → FIX FIRST before new work.**

## Code Style & Architecture

### Backend (FastAPI + Python)

**File Organization:**
```
backend/
├── app/
│   ├── api/           # Route handlers (thin)
│   ├── services/      # Business logic (thick)
│   ├── models/        # SQLAlchemy models
│   ├── schemas/       # Pydantic schemas
│   ├── core/          # Config, deps, security
│   └── main.py        # FastAPI app
├── tests/
│   ├── unit/          # Fast, isolated tests
│   ├── integration/   # DB + service tests
│   └── e2e/           # Full API tests
└── alembic/           # Database migrations
```

**Code Standards:**
- **Async everywhere**: All route handlers and DB calls use `async/await`
- **Type hints mandatory**: Every function has parameter and return types
- **Pydantic for validation**: Never manually validate input
- **Dependency injection**: Use FastAPI's `Depends()` for DB sessions, auth
- **Services not in routes**: Routes should be 5-10 lines max, call services

**Example Pattern:**
```python
# ❌ DON'T: Fat route handler
@router.post("/games")
async def create_game(game: GameCreate, db: Session = Depends(get_db)):
    # ... 50 lines of business logic ...
    pass

# ✅ DO: Thin route, thick service
@router.post("/games", response_model=GameResponse)
async def create_game(
    game: GameCreate,
    service: GameService = Depends(get_game_service)
) -> GameResponse:
    return await service.create_game(game)
```

**Testing Pattern:**
```python
# tests/unit/test_game_service.py
@pytest.mark.asyncio
async def test_create_game_records_result():
    # Given
    service = GameService(mock_db)
    game_data = GameCreate(player_id="123", score=100)
    
    # When
    result = await service.create_game(game_data)
    
    # Then
    assert result.score == 100
    assert result.recorded_at is not None
```

### Frontend (Next.js + TypeScript)

**File Organization:**
```
frontend/
├── src/
│   ├── app/              # Next.js App Router
│   │   ├── (routes)/     # Page components
│   │   └── api/          # API routes (if needed)
│   ├── components/       # React components
│   │   ├── ui/           # Reusable UI primitives
│   │   └── features/     # Feature-specific components
│   ├── lib/              # Utilities, API client
│   │   ├── api/          # Generated API client
│   │   └── hooks/        # Custom React hooks
│   └── types/            # Shared TypeScript types
└── __tests__/            # Jest/Vitest tests
```

**Code Standards:**
- **TypeScript strict mode**: No `any`, no type assertions without justification
- **Server components by default**: Only use 'use client' when needed
- **API types from OpenAPI**: Generate from backend's `/openapi.json`
- **Composition over props drilling**: Use context/hooks for shared state
- **Tailwind for styling**: No inline styles, use Tailwind classes

**Example Pattern:**
```typescript
// ❌ DON'T: Inline API calls
export default function Leaderboard() {
  const [data, setData] = useState([]);
  
  useEffect(() => {
    fetch('/api/rankings').then(r => r.json()).then(setData);
  }, []);
  
  return <div>{data.map(...)}</div>;
}

// ✅ DO: Use custom hook + type safety
export default function Leaderboard() {
  const { data, isLoading, error } = useRankings();
  
  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  
  return <RankingsList rankings={data} />;
}
```

## Git Workflow

### Commit Messages (Conventional Commits)
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code change that neither fixes nor adds
- `test`: Adding or updating tests
- `docs`: Documentation only
- `chore`: Tooling, dependencies, config

**Examples:**
```
feat(leaderboard): add real-time ranking updates

Implement WebSocket connection to stream ranking changes.
Users now see live updates without manual refresh.

Closes #123

---

fix(auth): prevent token refresh race condition

Add mutex lock around token refresh logic to prevent
concurrent refresh attempts that invalidate each other.

---

refactor(game-service): extract scoring logic to separate module

Move complex scoring calculations into dedicated service
for better testability and reusability.
```

### Branch Strategy
- `main`: Always deployable, protected
- `feature/description`: Short-lived feature branches
- No long-lived branches, no merge commits

**Workflow:**
```bash
# Start feature
git checkout -b feat/add-websocket-rankings

# Make changes, commit frequently
git add -p  # Stage selectively
git commit -m "feat(leaderboard): add WebSocket connection handler"

# Before pushing, rebase on main
git fetch origin
git rebase origin/main

# If conflicts, resolve and continue
git rebase --continue

# Push (force with lease for safety)
git push -f --force-with-lease origin feat/add-websocket-rankings
```

## When Adding New Features

Follow this sequence:

1. **Understand the requirement**
   - Read existing code in that area
   - Check for similar patterns in codebase
   - Review relevant tests

2. **Write tests first (TDD)**
   ```bash
   # Backend
   cd backend
   pytest tests/unit/test_new_feature.py -v --watch
   
   # Frontend
   cd frontend
   bun run test -- --watch test_new_feature
   ```

3. **Implement minimally**
   - Write simplest code that makes tests pass
   - No premature optimization
   - No speculative features

4. **Refactor**
   - Extract duplication
   - Improve naming
   - Add comments only where code can't be self-explanatory

5. **Integration check**
   ```bash
   # Run full test suite
   make test
   
   # Type check everything
   make type-check
   
   # Try it locally
   docker-compose up
   ```

6. **Document if needed**
   - Update API documentation
   - Add ADR if architectural decision
   - Update README if user-facing

7. **Commit atomically**
   ```bash
   git add <changed-files>
   git commit -m "feat(scope): clear description"
   ```

## When Fixing Bugs

1. **Write failing test that reproduces bug**
   ```python
   def test_ranking_handles_tied_scores():
       # This should pass but currently fails
       rankings = calculate_rankings([
           Player("A", score=100),
           Player("B", score=100),
       ])
       assert rankings[0].rank == 1
       assert rankings[1].rank == 1  # Not 2!
   ```

2. **Fix the bug**
   - Make the test pass
   - Don't touch unrelated code

3. **Verify fix doesn't break other tests**
   ```bash
   pytest -v  # All tests should still pass
   ```

4. **Commit with clear description**
   ```bash
   git commit -m "fix(rankings): handle tied scores correctly

   Players with identical scores now receive the same rank
   instead of sequential ranks."
   ```

## Dependencies

### Adding Dependencies

**ALWAYS justify in commit message:**
```bash
# Backend
cd backend
uv add package-name

git commit -m "chore(deps): add package-name for <reason>

We need this for <specific use case> because <justification>.
Alternatives considered: <list alternatives and why not chosen>."
```

**Questions to ask:**
- Is this well-maintained? (Check last commit date)
- Do we really need it? (Can we write it in 20 lines?)
- Does it have type stubs? (Python packages)
- What's the bundle size impact? (Frontend packages)

### Updating Dependencies

**Check for breaking changes first:**
```bash
# Backend
uv update --dry-run

# Frontend  
bun run outdated
```

**Update conservatively:**
- Minor/patch versions: Safe to update together
- Major versions: One at a time, test thoroughly

## Performance Considerations

### Backend
- Use database indexes for frequently queried fields
- Add `limit` to all list endpoints (default: 100)
- Use `select_related` / `joinedload` to avoid N+1 queries
- Cache expensive computations (Redis if needed)

### Frontend
- Use `loading.tsx` for instant loading states
- Implement pagination for long lists
- Lazy load images with `next/image`
- Use React Server Components for static content

## Security Checklist

Before committing code that touches:

**Authentication/Authorization:**
- [ ] No passwords or tokens in code
- [ ] Use environment variables
- [ ] Validate JWT on every protected endpoint
- [ ] Check user permissions, not just authentication

**Database:**
- [ ] Use parameterized queries (ORM handles this)
- [ ] Validate all input with Pydantic
- [ ] Sanitize user-generated content for XSS

**API:**
- [ ] Rate limiting on public endpoints
- [ ] CORS configured correctly
- [ ] No sensitive data in error messages

## What NOT to Do

### ❌ Never
- Commit directly to `main`
- Use `git merge` (rebase only)
- Skip tests "just this once"
- Use `any` type in TypeScript
- Use `# type: ignore` in Python (without comment explaining why)
- Add `console.log` without removing before commit
- Leave commented-out code
- Commit TODOs without GitHub issue reference

### ❌ Avoid
- Large commits (>300 lines changed)
- Long-running feature branches (>3 days)
- Mocking in integration tests
- Over-engineering simple features
- Premature abstraction

### ✅ Always
- Run tests before committing
- Keep commits atomic and focused
- Write descriptive commit messages
- Update tests when changing behavior
- Refactor immediately when you see duplication
- Ask questions when uncertain

## Special Notes for This Project

### Working with Existing Codebase
Since the backend is forked from an existing LLM agent codebase:

1. **Preserve upstream compatibility when possible**
   - Don't rename core modules unnecessarily
   - Keep similar file structure for easier merging
   - Document deviations in `./architecture-decisions/`

2. **Identify stable vs. evolving parts**
   - LLM agent core: Treat as relatively stable
   - Leaderboard logic: Our primary development area
   - API layer: Bridge between the two

3. **Test at integration boundaries**
   - Focus tests on how leaderboard code uses agent code
   - Mock external APIs, not internal services

## Getting Help

If uncertain about:
- **Architecture**: Check `./architecture-decisions/`
- **Patterns**: Look for similar existing code
- **Testing**: Check test files in same area
- **Dependencies**: Ask before adding

When in doubt, ask the human developer. Better to pause and clarify than to charge ahead with wrong assumptions.

## Success Metrics

You're doing well when:
- ✅ All tests pass on every commit
- ✅ Types check strictly with no errors
- ✅ Git history reads like a story
- ✅ Code reviews have minimal feedback
- ✅ Features deploy without incidents
- ✅ You can understand your own code 6 months later

Remember: **Code is read 10x more than it's written. Optimize for readability.**
