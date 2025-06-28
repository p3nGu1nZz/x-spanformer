import pytest


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item, call):
    """
    Attach a marker to each test report to identify tests that should not run in parallel.
    This allows for more granular control over test execution order.
    """
    outcome = yield
    report = outcome.get_result()
    if item.get_closest_marker("xdist_group"):
        report.xdist_group = item.get_closest_marker("xdist_group").args[0]
