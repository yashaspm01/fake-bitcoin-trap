import logging

from trap.timing import timed


def test_timed_keeps_result_and_name(caplog):
    @timed
    def add(a: int, b: int) -> int:
        return a + b

    with caplog.at_level(logging.INFO):
        assert add(1, 2) == 3
    assert add.__name__ == "add"
    assert "add took" in caplog.text
