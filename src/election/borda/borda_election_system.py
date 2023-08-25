import logging

from src.election.aggregation.simple_addition import SimpleAdditionVoteAggregation
from src.election.election_properties import ElectionProperties
from src.election.evaluation.point_limit_evaluation import PointThresholdEvaluation
from src.election.evaluation.simple_winner_evaluation import SimpleWinnerEvaluation
from src.util.point_vote import IllegalVoteException

log = logging.getLogger(__name__)


class Borda(ElectionProperties):

    aggregator = SimpleAdditionVoteAggregation

    def __init__(self, candidates, list_of_points=None, allow_less_cands=False, begin_with_last=False, allow_equality=False, expected_votes_n=None, num_winners=1, point_limit=None, **base_settings):
        """list_of_points: starting with the points of the winner, ending with the least points > 0 which
        | allow_less_cands: if false, exception will be raised if less candidates are ranked than the length of list_of_points
        | begin_with_last: effects only with allow_less_cands; example for false: 5,4,3,0,0,0; equivalent for true: 3,2,1,0,0,0
        | num_winners: this count of winners will be returned at minimum; in case of equality, more are possible
        | point_limit: all candidates who reached this limit will be returned as winner; if set, num_winners has no effect"""
        base_settings["logger_name_extension"] = str(list_of_points) + str(allow_less_cands) + str(begin_with_last) + str(allow_equality) + str(expected_votes_n) + '#' + str(num_winners) + '#' + str(point_limit)
        super().__init__(candidates, expected_votes_n=expected_votes_n, **base_settings)
        self.allow_rank_not_all = allow_less_cands
        self.allow_equality = allow_equality
        self.begin_with_last = begin_with_last
        self.list_of_winner_points = list_of_points if list_of_points is not None else [i for i in range(self.n_cand, 0, -1)]
        self.MAX_POINTS_PER_VOTE = self.list_of_winner_points[0]
        self.num_winners = num_winners
        self.point_limit = point_limit

    def serialize(self):
        borda_specific_settings = {
            "allowNotAllRanked": self.allow_rank_not_all,
            "beginWithLast": self.begin_with_last,
            "allowEquality": self.allow_equality,
            "numWinners": self.num_winners,
            "pointLimit": self.point_limit,
            "listOfWinnerPoints": self.list_of_winner_points
        }
        settings = super().serialize()
        settings["evaluation"].setdefault("settings", {}).update(borda_specific_settings)
        return settings

    @classmethod
    def serialized_to_args(cls, serialized):
        borda_specific_args = {
            "list_of_points": serialized["evaluation"]["settings"].get("listOfWinnerPoints"),
            "allow_less_cands": serialized["evaluation"]["settings"].get("allowNotAllRanked"),
            "begin_with_last": serialized["evaluation"]["settings"].get("beginWithLast"),
            "allow_equality": serialized["evaluation"]["settings"].get("allowEquality"),
            "num_winners": serialized["evaluation"]["settings"].get("numWinners"),
            "point_limit": serialized["evaluation"]["settings"].get("pointLimit")
        }
        args = super().serialized_to_args(serialized)
        args.update(borda_specific_args)
        return args

    def get_evaluator(self, n_votes, abb):
        if self.point_limit:
            return PointThresholdEvaluation(n_votes * self.MAX_POINTS_PER_VOTE, abb.enc_no_r(self.point_limit))
        else:
            return SimpleWinnerEvaluation(abb.get_bits_for_size(n_votes * self.MAX_POINTS_PER_VOTE), self.num_winners)

    def generate_valid_vote(self, generic_vote, abb):
        vote = {}
        vote_decrypted = {}

        number_of_ranked_cands = self.n_cand - generic_vote.get_number_of_ignored()

        if not self.allow_rank_not_all and number_of_ranked_cands < len(self.list_of_winner_points):
            raise IllegalVoteException("More candidates have to be ranked")

        if not self.allow_equality and generic_vote.get_first_doubled_position() > 0:
            raise IllegalVoteException("This Borda doesn't support doubled positions")

        for cand in range(self.n_cand):
            vote_decrypted[cand] = 0

        # use offset if begin_with_last is enabled and not enough candidates are ranked
        offset = 0
        if self.begin_with_last and len(self.list_of_winner_points) > self.n_cand - generic_vote.get_number_of_ignored():
            offset = len(self.list_of_winner_points) - (self.n_cand - generic_vote.get_number_of_ignored())

        for position, candidates in generic_vote.get_ranking_map().items():
            if position - 1 < len(self.list_of_winner_points) and position > 0:
                for cand in candidates:
                    vote_decrypted[cand] = self.list_of_winner_points[position - 1 + offset]

        log.debug(str(vote_decrypted))
        for cand in range(self.n_cand):
            vote[cand] = abb.enc(vote_decrypted[cand])

        return vote

    def serialize_vote(self, valid_vote):
        serialized_vote = {"choices": [
            vote.serialize() for cand, vote in valid_vote.items()
        ]}
        return serialized_vote

    def deserialize_vote(self, serialized_vote, abb):
        deserialized_vote = [
            abb.deserialize_cipher(vote) for vote in serialized_vote["choices"]
        ]
        return deserialized_vote
