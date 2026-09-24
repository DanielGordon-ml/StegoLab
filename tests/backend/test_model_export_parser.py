"""Keep mistaken secret arguments out of standalone runtime parser output."""

import pytest

from backend_service.model_export_runtime import main


@pytest.mark.parametrize(
    "arguments",
    [
        ["decode", "--password", "public-review-sentinel"],
        ["public-review-sentinel"],
        ["encode", "--message=public-review-sentinel"],
    ],
)
def test_export_parser_does_not_echo_secret_arguments(
    arguments: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject unknown options and invalid actions before any graph is loaded."""
    with pytest.raises(SystemExit) as failure:
        main(arguments)
    captured = capsys.readouterr()
    assert failure.value.code == 2
    assert captured.out == ""
    assert captured.err == (
        "The model command arguments are invalid. Use --help for examples.\n"
    )
    assert "public-review-sentinel" not in captured.out + captured.err
