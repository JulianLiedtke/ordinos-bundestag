from src.protocols.protocol import Protocol


class RetrieveVotesProtocol(Protocol):

    def __init__(self, bulletin_board_client, deserialize_vote):
        super().__init__()
        self.bulletin_board_client = bulletin_board_client
        self.deserialize_vote = deserialize_vote

    def run(self):
        ballots = self.bulletin_board_client.get_votes()
        return [[self.deserialize_vote(ballot) for ballot in ballots]]
        # return [ballots]
