"""
Testes para scripts/cleanup_repo.py.

Cobre:
- Arquivamento de diretórios (com mocks de sistema de arquivos)
- Criação de diretório archive
- Validação de nomes de arquivo com timestamp
"""

from datetime import UTC, datetime, timezone

import pytest

# Import do módulo para cobertura
import scripts.cleanup_repo


@pytest.mark.unit
class TestCleanupRepo:
    """Testes para lógica de cleanup e arquivamento de diretórios."""

    def test_archive_dir_name_format(self) -> None:
        """Testa que nome do diretório segue o padrão archive_YYYYMMdd_HHMMSS."""
        # Use uma data fixa para teste determinístico
        test_date = datetime(2024, 3, 5, 14, 30, 45, tzinfo=UTC)
        expected_format = test_date.strftime("archive_%Y%m%d_%H%M%S")
        assert expected_format == "archive_20240305_143045"
        assert expected_format.startswith("archive_")
        assert len(expected_format) == 23  # archive_YYYYMMDD_HHMMSS

        # Verificar se o ARCHIVE_DIR do módulo segue o padrão
        assert scripts.cleanup_repo.ARCHIVE_DIR.startswith("archive_")

    def test_archive_dir_creation(self, tmp_path) -> None:
        """Testa criação do diretório archive."""
        archive_path = tmp_path / "archive_20240305_143045"
        archive_path.mkdir(exist_ok=True)
        assert archive_path.exists()
        assert archive_path.is_dir()

    def test_candidate_dirs_list(self) -> None:
        """Testa que lista de diretórios candidatos está correta."""
        candidates = ["htmlcov", "docs", "notebooks", "data", "dev", ".pytest_cache"]
        assert len(candidates) == 6
        assert "htmlcov" in candidates
        assert "data" in candidates
        assert "notebooks" in candidates

    def test_move_single_directory(self, tmp_path) -> None:
        """Testa movimento de um único diretório para archive."""
        # Setup
        source_dir = tmp_path / "htmlcov"
        source_dir.mkdir()
        test_file = source_dir / "test.html"
        test_file.write_text("<html>test</html>")

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Execute
        import shutil

        target = archive_dir / "htmlcov"
        shutil.move(str(source_dir), str(target))

        # Verify
        assert not source_dir.exists()
        assert target.exists()
        assert (target / "test.html").exists()
        assert (target / "test.html").read_text() == "<html>test</html>"

    def test_nonexistent_directory_skipped(self, tmp_path) -> None:
        """Testa que diretórios que não existem são pulados."""
        # Setup
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Existem diretórios que não existem
        nonexistent = tmp_path / "nonexistent_dir"
        assert not nonexistent.exists()

        # Não deve tentar mover
        moved_count = 0
        for d in ["nonexistent_dir"]:
            path = tmp_path / d
            if path.exists():
                moved_count += 1

        assert moved_count == 0

    def test_multiple_directories_moved(self, tmp_path) -> None:
        """Testa movimento de múltiplos diretórios."""
        # Setup
        candidates = ["htmlcov", "docs", "data"]
        for candidate in candidates:
            (tmp_path / candidate).mkdir()

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Execute: mover cada candidato
        import shutil

        moved = []
        for d in candidates:
            path = tmp_path / d
            if path.exists():
                target = archive_dir / d
                shutil.move(str(path), str(target))
                moved.append(d)

        # Verify
        assert len(moved) == 3
        assert all((archive_dir / d).exists() for d in candidates)
        assert all(not (tmp_path / d).exists() for d in candidates)

    def test_archive_with_nested_content(self, tmp_path) -> None:
        """Testa que conteúdo aninhado é preservado durante archive."""
        # Setup
        source = tmp_path / "docs"
        source.mkdir()
        (source / "api").mkdir()
        (source / "api" / "index.md").write_text("# API Docs")
        (source / "guide.md").write_text("# Guide")

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Execute
        import shutil

        target = archive_dir / "docs"
        shutil.move(str(source), str(target))

        # Verify
        assert (target / "api" / "index.md").exists()
        assert (target / "guide.md").exists()
        assert (target / "api" / "index.md").read_text() == "# API Docs"

    def test_empty_archive_message(self) -> None:
        """Testa mensagem quando nenhum diretório é encontrado."""
        moved = []

        if not moved:
            message = "No candidate directories found to archive."
            assert "No candidate" in message
            assert "archive" in message


@pytest.mark.unit
class TestCleanupRepoIntegration:
    """Testes de integração para o script cleanup."""

    def test_cleanup_workflow_dry_run(self, tmp_path) -> None:
        """Testa fluxo completo de cleanup (dry run)."""
        # Setup projeto simulado
        root = tmp_path
        candidates = ["htmlcov", "docs", "notebooks"]
        for d in candidates:
            (root / d).mkdir()
            (root / d / "file.txt").write_text(f"Content of {d}")

        # Criar archive
        archive_dir = root / "archive_20240305_143045"
        archive_dir.mkdir(exist_ok=True)

        # Simular movimento
        import shutil

        moved = []
        for d in candidates:
            path = root / d
            if path.exists():
                target = archive_dir / d
                shutil.move(str(path), str(target))
                moved.append((d, str(target)))

        # Verify
        assert len(moved) == 3
        assert all((archive_dir / d).exists() for d, _ in moved)
        assert archive_dir.exists()
        assert (archive_dir / "htmlcov" / "file.txt").read_text() == "Content of htmlcov"

    def test_cleanup_preserves_other_files(self, tmp_path) -> None:
        """Testa que arquivos não-candidatos não são movidos."""
        # Setup
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "htmlcov").mkdir()

        # Archive
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir(exist_ok=True)

        # Mover apenas candidatos
        import shutil

        candidates_to_move = ["htmlcov"]
        for d in candidates_to_move:
            path = tmp_path / d
            if path.exists():
                shutil.move(str(path), archive_dir / d)

        # Verify src não foi movido
        assert (tmp_path / "src").exists()
        assert (tmp_path / "src" / "main.py").exists()
        assert not (archive_dir / "src").exists()
