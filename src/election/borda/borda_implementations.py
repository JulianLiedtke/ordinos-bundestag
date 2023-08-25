from src.election.borda.borda_election_system import Borda
from src.election.election_properties import ElectionProperties

class EscElection(Borda):
    """https://de.wikipedia.org/wiki/Eurovision_Song_Contest"""
    def __init__(self, candidates, **base_settings):
        base_settings.update({
            "list_of_points": [12, 10, 8, 7, 6, 5, 4, 3, 2, 1],
            "allow_less_cands": True
        })
        super().__init__(candidates, **base_settings)

class MedalTableSystem(Borda):
    """https://de.wikipedia.org/wiki/Medaillenspiegel"""
    def __init__(self, candidates, count_of_comptetions, **base_settings):
        base_settings.update({
            "list_of_points": [pow(count_of_comptetions + 1, 2) + 1, count_of_comptetions + 1, 1],
            "expected_votes_n": count_of_comptetions
        })
        super().__init__(candidates, **base_settings)

    @classmethod
    def serialized_to_args(cls, serialized):
        args = ElectionProperties.serialized_to_args(serialized)
        args.update({"candidates": args["candidates"],
                     "count_of_comptetions": args["expected_votes_n"]})
        return args

class FisWorldCup(Borda):
    """https://de.wikipedia.org/wiki/FIS-Punktesystem"""
    def __init__(self, candidates, count_of_comptetions, **base_settings):
        base_settings.update({
            "list_of_points": [100, 80, 60, 50, 45, 40, 36, 32, 29, 26, 24, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            "expected_votes_n": count_of_comptetions,
            "allow_less_cands": True,
            "allow_equality": True
        })
        super().__init__(candidates, **base_settings)

    @classmethod
    def serialized_to_args(cls, serialized):
        args = ElectionProperties.serialized_to_args(serialized)
        args.update({"candidates": args["candidates"],
                     "count_of_comptetions": args["expected_votes_n"]})
        return args
