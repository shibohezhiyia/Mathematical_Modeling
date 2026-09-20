"""The development probe keeps hidden generator labels out of solver inputs."""
from scripts.run_independent_generator_decision_probe import _cases


def test_independent_generator_probe_is_deterministic_and_has_disjoint_queries():
    first = list(_cases(20260920))
    second = list(_cases(20260920))
    assert [name for name, _, _ in first] == [
        'sklearn_friedman2', 'sklearn_friedman3', 'sklearn_linear3',
    ]
    for (name, payload, reference), (_, repeated, repeated_reference) in zip(first, second):
        assert payload == repeated
        assert reference.tolist() == repeated_reference.tolist()
        rows = payload['attachments'][0]['rows']
        queries = payload['query_inputs']
        assert len(rows) == 192
        assert len(queries) == len(reference) == 64
        assert all('response' in row for row in rows)
        assert all(isinstance(query, list) and len(query) == len(rows[0]) - 1 for query in queries)
        columns = sorted(set(rows[0]) - {'response'})
        train_inputs = {tuple(row[column] for column in columns) for row in rows}
        assert train_inputs.isdisjoint(map(tuple, queries))
        assert 'hidden_reference' not in payload
