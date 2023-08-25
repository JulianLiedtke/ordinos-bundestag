import logging
from src.election.aggregation.simple_addition import SimpleAdditionVoteAggregation

from src.election.election_properties import ElectionProperties
from src.election.evaluation.simple_winner_evaluation import \
    SimpleWinnerEvaluation
from src.util.point_vote import IllegalVoteException

log = logging.getLogger(__name__)


class SingleVoteElection(ElectionProperties):

    aggregator = SimpleAdditionVoteAggregation

    def __init__(self, candidates, **base_settings):
        super().__init__(candidates, **base_settings)
        self.MAX_POINTS_PER_VOTE = 1

    def generate_valid_vote(self, generic_vote, abb):
        """ inits a dictionary containing each candidate with zero votes """
        vote_decrypted = {}

        # set every cand to 0 and find cand with biggest points, prefer lower numbers if equals
        for cand in range(self.n_cand):
            vote_decrypted[cand] = 0

        # set index with highest points to 1
        winners = generic_vote.get_position(1)
        if len(winners) != 1:
            raise IllegalVoteException("More or less than one winner")
        vote_decrypted[winners[0]] = 1

        vote_encrypted = {}
        log.debug(str(vote_decrypted))
        for cand in range(self.n_cand):
            vote_encrypted[cand] = abb.enc(vote_decrypted[cand])
        return vote_encrypted

    def get_evaluator(self, n_votes, abb):
        return SimpleWinnerEvaluation(abb.get_bits_for_size(n_votes * self.MAX_POINTS_PER_VOTE))

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
