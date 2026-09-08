"""Basic CLI availability without optional UI/model dependencies."""


def test_cli_entry_is_callable():
    from credit_one import run

    assert callable(run.main)


def test_help_lists_available_commands(capsys):
    from credit_one import run

    assert run.main([]) == 2
    output = capsys.readouterr().out
    assert "demo" in output
    assert "validate" in output
    assert "dashboard" in output
