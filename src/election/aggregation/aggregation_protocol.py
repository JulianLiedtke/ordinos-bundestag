from src.protocols.protocol import Protocol


class AggregationProtocol(Protocol):

    def __init__(self, aggregator, abb, n_cands):
        self.aggregator = aggregator
        self.abb = abb
        self.n_cands = n_cands
        super().__init__()

    def run(self, votes):
        aggregate_votes = self.aggregator.get_initial_vote_aggregation(self.abb, self.n_cands)
        for vote in votes:
            self.aggregator.aggregate_vote(aggregate_votes, vote)
        return [aggregate_votes]