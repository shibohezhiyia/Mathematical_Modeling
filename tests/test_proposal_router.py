import pytest

from core.proposal_router import BackendProfile, ProposalRouterError, route_proposal_backends


class Backend:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_router_prefers_declared_quality_then_requires_shared_validator():
    slow = Backend('{"model":"slow"}')
    fast = Backend('{"model":"fast"}')
    profiles = [BackendProfile("slow", slow, .9, .8, 100, 2),
                BackendProfile("fast", fast, .9, .8, 10, 1)]
    result = route_proposal_backends(profiles, [{"role": "user", "content": "x"}],
                                     validator=lambda text: {"status": "accepted"})
    assert result["status"] == "accepted" and result["backend"] == "fast"
    assert slow.calls == 0 and fast.calls == 1


def test_router_keeps_failed_and_rejected_backends_bounded():
    broken = Backend(RuntimeError("offline"))
    rejected = Backend("bad")
    result = route_proposal_backends(
        [BackendProfile("broken", broken, .9, .1, 1, 1), BackendProfile("rejected", rejected, .8, .1, 1, 1)],
        [], validator=lambda text: {"status": "rejected", "diagnostic": "schema"}, max_calls=2)
    assert result["status"] == "not_assessed"
    assert len(result["attempts"]) == 2


def test_router_rejects_invalid_profiles_and_budgets():
    with pytest.raises(ProposalRouterError, match="rate"):
        BackendProfile("x", Backend("x"), 2, 0, 1, 1)
    with pytest.raises(ProposalRouterError, match="total_call"):
        route_proposal_backends([BackendProfile("x", Backend("x"), 0, 0, 1, 1)], [],
                                validator=lambda text: {"status": "rejected"}, max_calls=True)
