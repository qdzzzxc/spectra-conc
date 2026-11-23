"""Проверки общего источника проектных скиллов Claude Code и Codex."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE_SKILLS = ROOT / ".claude" / "skills"
CODEX_SKILLS = ROOT / ".agents" / "skills"


def test_skill_folders_match() -> None:
    """Обнаруживает пропущенные, лишние и переименованные скиллы."""
    claude_names = {entry.name for entry in CLAUDE_SKILLS.iterdir() if not entry.name.startswith(".")}
    codex_names = {entry.name for entry in CODEX_SKILLS.iterdir() if not entry.name.startswith(".")}

    assert claude_names, "В .claude/skills нет проектных скиллов"
    assert claude_names == codex_names, (
        f"Нет в Codex: {sorted(claude_names - codex_names)}. "
        f"Лишние в Codex: {sorted(codex_names - claude_names)}."
    )


def test_codex_skills_point_to_claude_folders() -> None:
    """Обнаруживает копии, битые ссылки и ссылки на другую папку."""
    for source in sorted(CLAUDE_SKILLS.iterdir()):
        if source.name.startswith("."):
            continue
        link = CODEX_SKILLS / source.name
        assert (source / "SKILL.md").is_file(), f"Нет SKILL.md в {source}"
        assert link.is_symlink(), f"{link} должен быть симлинком"
        assert not link.readlink().is_absolute(), f"{link} должен быть относительным"
        assert link.is_dir() and link.resolve() == source.resolve(), (
            f"{link} должен указывать на {source}"
        )
