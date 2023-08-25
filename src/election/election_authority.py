

import base64
import logging
import hashlib
import secrets
import json
from src import election
from src.election.election_properties import ElectionProperties
from time import time

from src.election.bulletin_board.bulletin_board import BulletinBoard
from src.election.bulletin_board.bulletin_board_client import BulletinBoardClient
from src.network.retrieve_votes_protocol import RetrieveVotesProtocol
from src.util.csv_writer import CSV_Writer
from src.util.point_vote import IllegalVoteException
from src.util.protocol_chain import ChainedProtocol
from src.util.protocol_runner import ProtocolRunner
from src.election.aggregation.aggregation_protocol import AggregationProtocol
from src.election.tie_breaking.tie_breaking import TieBreaking, BreakingTypes
import logging
log = logging.getLogger(__name__)


class ElectionAuthority():

    def __init__(self, trustee_gen, election_properties: ElectionProperties, local_bulletin_board=False):
        self.election_system_name = election_properties.system_name
        self.logger = logging.getLogger(self.election_system_name)

        self.n_cand = election_properties.n_cand
        self.candidate_names = election_properties.candidate_names
        self.use_constituencies = True if election_properties.use_constituencies else False
        self.n_votes = 0
        self.expected_votes_n = election_properties.expected_votes_n
        self.vote_transformation_time = 0

        self.serialize_vote = election_properties.serialize_vote
        self.deserialize_vote = lambda serialized: election_properties.deserialize_vote(serialized, self.abb)

        self.bulletin_board = BulletinBoard() if local_bulletin_board else BulletinBoardClient()

        trustees = trustee_gen(self.bulletin_board)
        self.abb = trustees[0].abb.create_local_abb()
        self.abb_log = trustees[0].abb.op_logger

        # tie breaking setup
        election_properties.setup(self.abb)
        

        self.generate_election_configs(election_properties)
        
        evaluation_protocol = lambda: ChainedProtocol([RetrieveVotesProtocol(self.bulletin_board, self.deserialize_vote),
                                                       # TODO FilterProtocol: only newest votes, with valid ZKP
                                                       AggregationProtocol(election_properties.aggregator, self.abb, self.n_cand),
                                                       election_properties.get_evaluator(self.n_votes, self.abb)])
        self.evaluation = ProtocolRunner(trustees, evaluation_protocol)
        self.generate_encrypted_vote = election_properties.generate_valid_vote

    def generate_ballots(self, n_voters, IP):
        """
        Generate all the secrets that allow the users to vote during an election, add them to
        the bulletin board and output them into the secrets folder. Only valid for remote
        bulletin boards, as the local versions do not have the address / board_id.

        Args:
            n_voters (int): The amount of secrets to generate, i.e. how many participants
                will this election have.
            IP (str): The base url of the bulletin board.

        Raises:
            TypeError: If this function is called when the bulletin_board is not an 
                instance of the BulletinBoardClient class.
        """
        if not isinstance(self.bulletin_board, BulletinBoardClient):
            raise TypeError("Ballot generation is only supported for remote bulletin boards.")

        secret_list = []
        for i in range(n_voters):
            # generate different secrets
            while True:
                secret = secrets.token_bytes(128)
                if secret not in secret_list:
                    break
            secret_list.append(secret)
            h = hashlib.sha512()
            h.update(secret)
            secret_hex = base64.b16encode(secret).decode("utf-8")
            hash_hex = h.hexdigest()
            self.logger.info("Adding secret %.8s... with hash %.8s...", secret_hex, hash_hex)
            self.bulletin_board.add_hash(hash_hex)
            participation_conformation = {
                "config_adress": f"{IP}/api/getConfig?board_id={self.bulletin_board.board_id}",
                "secret": secret_hex
            }
            with open('secrets/secret-' + str(i) + '.json', 'w') as f:
                json.dump(participation_conformation, f)

    def generate_election_configs(self, election_properties):
        """
        Publishes the passed election properties in serialized form on the bulletin board.

        Args:
            election_properties (ElectionProperties): The election properties that shall be
                published on the bulletin board.
        """
        serialized_properties = election_properties.serialize()
        
        # Only if the bulletin board is a remote bulletin board, we may add its id 
        if isinstance(self.bulletin_board, BulletinBoardClient):
            serialized_properties.setdefault("election", {}).update({"id": str(self.bulletin_board.board_id)})
        
        # Add the data for encryption of the votes
        serialized_properties.setdefault("encryption", {}).update({
                "bits": self.abb.pk.n.bit_length(),
                "publicKey": str(int(self.abb.pk.n))
            })
        self.logger.info("Adding election to bulletin board with properties: %s", serialized_properties)
        self.bulletin_board.set_election_config(serialized_properties)

    def add_votes_and_evaluate(self, votes):
        """
        send votes to the bulletin board and evaluate
        """
        valid_votes = self.__transform_votes(votes)
        self.__add_votes_to_bulletin_board(valid_votes)
        return self.trigger_evaluation()

    def add_generic_vote(self, vote, count=1):
        """
        send a vote to the bulletin board
        | vote could be added more than once, which could be specified by count
        """
        valid_votes = self.__transform_votes([vote] * count)
        self.__add_votes_to_bulletin_board(valid_votes)

    def __add_votes_to_bulletin_board(self, valid_votes):
        for valid_vote in valid_votes:
            if self.expected_votes_n != None and self.n_votes + 1 > self.expected_votes_n:
                self.logger.warning("Maximum number (" + str(self.expected_votes_n) + ") of votes reached. Vote couldn't be added.")
            else:
                serialized_vote = self.serialize_vote(valid_vote)
                # To avoid getting filtered out, pretend to have a unique hash, trustees currently don't check if hash is actually valid.
                serialized_vote["hash"] = self.n_votes
                self.bulletin_board.add_vote(serialized_vote)
                self.n_votes += 1

    def __transform_votes(self, generic_votes):
        start_time = time()

        valid_votes = []
        for vote in generic_votes:
            try:
                valid_vote = self.generate_encrypted_vote(vote, self.abb)
                valid_votes.append(valid_vote)
            except IllegalVoteException as e:
                self.logger.warning(str(vote) + " could't be added, because " + str(e))

        self.vote_transformation_time += time() - start_time
        return valid_votes

    def trigger_evaluation(self):
        self.logger.info('Computation time to transform votes: {:.3f}s'.format(self.vote_transformation_time))
        ballots = self.bulletin_board.get_votes()
        # TODO move these to where aggregation happens
        # self.logger.info('Computation time to aggregate votes: {:.3f}s'.format(aggregation_time))
        # CSV_Writer.set_eval_time(aggregation_time)
        self.n_votes = len(ballots)

        if self.n_votes == 0:
            self.logger.warning("Election couldn't be started, because no valid vote was added")
            return

        if self.expected_votes_n and self.n_votes != self.expected_votes_n:
            self.logger.warning("Election expected " + str(self.expected_votes_n) + ", but got " + str(self.n_votes) + " votes")

        self.logger.info("Start evaluation")

        startTime = time()
        result, eval_prot = self.evaluation.run()

        if result == -1 or result == 'Abort.':
            self.logger.error("Something went wrong in evaluation, result was {}".format(result))
            raise ValueError("Something went wrong in evaluation, result was {}".format(result))

        # find real winner name
        if self.use_constituencies:
            # get one winner for each constituency
            n_const = len(result)
            winner_names = [[] for i in range(n_const)]
            for i in range(n_const):
                for cand_index in result[i]:
                    winner_names[i].append(cand_index)
        else: 
            winner_names = []
            for cand_index in result:
                winner_names.append(self.candidate_names[cand_index])

        self.logger.info('Result: {}'.format(winner_names))

        t = time() - startTime
        # TODO make these stats accessible again
        gt_ops = self.abb_log.get_count_gt_operations()
        eq_ops = self.abb_log.get_count_eq_operations()
        dec_ops = self.abb_log.get_count_dec_operations()
        mul_ops = self.abb_log.get_count_mul_operations()
        #self.logger.info('Computation time to evaluate winner: {:.3f}s, with {} gt-ops, {} eq-ops, {} dec-ops and {} mul-ops'.format(t, gt_ops, eq_ops, dec_ops, mul_ops))
        #CSV_Writer.set_eval_time(t)
        #CSV_Writer.write_with_election_params(self.n_cand, self.n_votes, None, None, self.election_system_name, winner_names, gt_ops, eq_ops, dec_ops, mul_ops)
        return result
