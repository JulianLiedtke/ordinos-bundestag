from src.protocols.protocol import Protocol


class ConstituencyAggregationProtocol(Protocol):

    def __init__(self, aggregator, abb, n_cands_per_const):
        self.aggregator = aggregator
        self.abb = abb
        self.n_cands_per_const = n_cands_per_const
        super().__init__()

    def run(self, votes):
        aggregate_votes = self.aggregator.get_initial_vote_aggregation(self.abb, self.n_cands_per_const)
        for vote in votes:
            self.aggregator.aggregate_vote(aggregate_votes, vote)
        return [aggregate_votes]