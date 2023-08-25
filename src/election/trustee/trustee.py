from src.election.aggregation.aggregation_protocol import AggregationProtocol
from src.election.aggregation.constituency_aggregation_protocoll import ConstituencyAggregationProtocol
from src.election.election_properties import ElectionProperties
from src.election.bulletin_board.bulletin_board_client import BulletinBoardClient
from src.election.real_world_elections.bundestagelection import BundestagElection
from src.election.votes.one_candidate_vote import VoteType
from src.network.bulletin_board_connector import BulletinBoardChannel, BulletinBoardConnector
from src.network.local_connector import LocalConnector
from src.network.connection import Channel
from threading import Thread
import logging

from src.network.retrieve_votes_protocol import RetrieveVotesProtocol
from src.util.protocol_chain import ChainedProtocol
from src.util.protocol_parallel import ParallelProtocol

log = logging.getLogger(__name__)


class Trustee():

    def __init__(self, abb, id, bulletin_board, channel):
        self.run_finished = False

        self.id = id
        self.abb = abb
        self.bulletin_board = bulletin_board
        self.channel = channel
        self.result = None

    def run_protocol(self, prot, *args):
        self.run_finished = False

        self.abb.prot_suite.set_connections(self.channel)
        prot.set_connection(self.channel)
        prot.set_abb(self.abb)
        prot.set_output_func(self.receive_prot_output)
        self.protocol = prot

        self.thread = Thread(target=prot.start, args=args)
        self.thread.start()

    def receive_prot_output(self, result):
        self.run_finished = True
        self.result = result
        self.bulletin_board.add_result(self.channel.id, result)

    def is_protocol_finished(self):
        if self.run_finished:
            return True
        if not self.thread.is_alive():
            if not self.result:
                self.result = "Abort."
            return True
        return False

    def get_protocol(self):
        return self.protocol

    def trigger_evaluation(self):
        """
        Begins the evaluation of the election in another thread.
        Would be more like the original trustee if we could use the chained protocol,
        and simply call run protocoll instead.
        """
        self.run_finished = False
        self.thread = Thread(target=self.evaluate)
        self.thread.start()

    def evaluate(self):
        """
        Actual evaluation of the election, is run in thread. It runs following steps:
        1. Retrieve votes: Get all the votes listed on the bulletin board.
        2. Filter votes: Filter out only those with valid ZKP, and only the newest per hash.
        3. Aggregate: Aggregate according to the vote structure.
        4. Evaluate: Run the actual evaluation algorithm specified by the election properties.
        """
        election_properties: ElectionProperties = self.bulletin_board.get_election_config()

        log.info("Trustee [%s] begins evaluation on BB %s as id %s with partners %s",
                 self.id, self.bulletin_board.board_id, self.channel.id, self.channel.other_trustee_ids)

        # 1. Retrieve the votes
        votes = self.bulletin_board.get_votes()

        # more complex evaluation because two types of votes (primary votes and secondary votes) are used and each type has different constituencies etc.
        if type(election_properties) == BundestagElection:
            # 2. Filter the votes
            # TODO: primary and secondary votes do not necessary have the same constituencies
            valid_primary_votes = [{} for i in range(election_properties.n_constituencies)]
            valid_secondary_votes = [{} for i in range(election_properties.n_constituencies)]
            # n_votes are separated by constituencies
            n_valid_primary_votes = [0 for i in range(election_properties.n_constituencies)]
            n_valid_secondary_votes = [0 for i in range(election_properties.n_constituencies)]

            for vote in votes:
               # log.info(vote)
                if "hash" in vote and "constituency" in vote and "type" in vote:
                    if vote["type"] == VoteType.first_vote:
                        if (vote["hash"] not in valid_primary_votes or vote["timestamp"] > valid_primary_votes[vote["hash"]]["timestamp"]):
                            # sort valid votes per constituency
                            valid_primary_votes[vote["constituency"]][vote["hash"]] = vote
                            n_valid_primary_votes[vote["constituency"]] += 1
                    elif vote["type"] == VoteType.secondary_vote:
                        if (vote["hash"] not in valid_secondary_votes or vote["timestamp"] > valid_secondary_votes[vote["hash"]]["timestamp"]):
                            # sort valid votes per constituency
                            valid_secondary_votes[vote["constituency"]][vote["hash"]] = vote
                            n_valid_secondary_votes[vote["constituency"]] += 1
                
                
            valid_primary_votes = [list(valid_primary_votes[i].values()) for i in range(len(valid_primary_votes))]
            valid_secondary_votes = [list(valid_secondary_votes[i].values()) for i in range(len(valid_secondary_votes))]

            log.info("Found %s/%s primary votes with unique hash for the evaluation.", sum(n_valid_primary_votes), len(votes))
            log.info("Found %s/%s secondary votes with unique hash for the evaluation.", sum(n_valid_secondary_votes), len(votes))

            # TODO 2.1 Filter by passing and failing ZKP

            # Convert them to the votes used by the evaluation algorithms (no extra meta-data like hash, timestamp etc)
            for const_votes in range(len(valid_primary_votes)):
                valid_primary_votes[const_votes] = [election_properties.get_choices(vote, self.abb) for vote in valid_primary_votes[const_votes]]
            for const_votes in range(len(valid_secondary_votes)):
                valid_secondary_votes[const_votes] = [election_properties.get_choices(vote, self.abb) for vote in valid_secondary_votes[const_votes]]
            
            # 3+4. Aggregate and evaluate the votes
            single_aggregation_protocol = AggregationProtocol(election_properties.aggregator(), self.abb, election_properties.n_cand)
            aggregation_protocol = ParallelProtocol(ParallelProtocol(single_aggregation_protocol))
            # return n_valid_votes as a list with divided by vote type and constituency
            evaluation_protocol = ChainedProtocol([aggregation_protocol, election_properties.get_evaluator([n_valid_primary_votes, n_valid_secondary_votes], self.abb)])
            self.run_protocol(evaluation_protocol, [valid_primary_votes, valid_secondary_votes])

        # The following part of the evaluation depents whether constituencies are used
        elif(election_properties.use_constituencies):
            # 2. Filter the votes
            valid_votes = [{} for i in range(election_properties.n_constituencies)]
            n_valid_votes = [0 for i in range(election_properties.n_constituencies)]
            for vote in votes:
                if "hash" in vote and "constituency" in vote:
                    if (vote["hash"] not in valid_votes or vote["timestamp"] > valid_votes[vote["hash"]]["timestamp"]):
                        # sort valid votes per constituency
                        valid_votes[vote["constituency"]][vote["hash"]] = vote
                        n_valid_votes[vote["constituency"]] += 1
                
                
            valid_votes = [list(valid_votes[i].values()) for i in range(len(valid_votes))]
            log.info("Found %s/%s votes with unique hash for the evaluation.", sum(n_valid_votes), len(votes))

            # TODO 2.1 Filter by passing and failing ZKP

            # Convert them to the votes used by the evaluation algorithms (no extra meta-data like hash, timestamp etc)
            for const_votes in range(len(valid_votes)):
                valid_votes[const_votes] = [election_properties.get_choices(vote, self.abb) for vote in valid_votes[const_votes]]
            
            # 3+4. Aggregate and evaluate the votes
            single_aggregation_protocol = []
            for i in range(election_properties.n_constituencies):
                single_aggregation_protocol.append(AggregationProtocol(election_properties.aggregator(), self.abb, election_properties.n_cand_per_const[i]))
            aggregation_protocol = ParallelProtocol(single_aggregation_protocol)
            evaluation_protocol = ChainedProtocol([aggregation_protocol, election_properties.get_evaluator(n_valid_votes, self.abb)])
            self.run_protocol(evaluation_protocol, valid_votes)


        else:
            # evaluation without constituencies
            # 2. Filter the votes
            valid_votes = {}
            for vote in votes:
                if "hash" in vote:
                    if (vote["hash"] not in valid_votes or
                        vote["timestamp"] > valid_votes[vote["hash"]]["timestamp"]):
                        valid_votes[vote["hash"]] = vote
            
            valid_votes = list(valid_votes.values())
            log.info("Found %s/%s votes with unique hash for the evaluation.", len(valid_votes), len(votes))

            # TODO 2.1 Filter by passing and failing ZKP

            # Convert them to the votes used by the evaluation algorithms (no extra meta-data like hash, timestamp etc)
            deserialize_vote = lambda seralized_vote: election_properties.deserialize_vote(seralized_vote, self.abb)
            valid_votes = [deserialize_vote(vote) for vote in valid_votes]

            # 3+4. Aggregate and evaluate the votes
            evaluation_protocol = ChainedProtocol([AggregationProtocol(election_properties.aggregator, self.abb, election_properties.n_cand),
                                                election_properties.get_evaluator(len(valid_votes), self.abb)])
            self.run_protocol(evaluation_protocol, valid_votes)


def init_trustees(bulletin_board, abbs, ids):
    trustees = []
    n_trustees = len(ids)

    log.debug("Creating trustees with ids : '%s'", ids)

    # create trustees
    for i in range(n_trustees):
        other_ids = ids.copy()
        other_ids.pop(i)
        channel = BulletinBoardChannel(bulletin_board, ids[i], other_ids)
        trustee = Trustee(abbs[i], ids[i], bulletin_board, channel)
        trustees.append(trustee)

    return trustees
