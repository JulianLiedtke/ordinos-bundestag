import logging

from src.election.aggregation.base_aggregation import \
    BaseVoteAggregation

log = logging.getLogger(__name__)


class VoteRememberVoteAggregation(BaseVoteAggregation):

    @staticmethod
    def get_initial_vote_aggregation(abb, n_cand):
        """ inits a dictionary containing each candidate with zero votes """
        return []

    @staticmethod
    def aggregate_vote(vote_aggregation, new_vote):
        vote_aggregation.append(new_vote)
