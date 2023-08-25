import logging
import hashlib
import base64
from datetime import date, datetime, timezone
# TODO: Maybe save configs/paramter in a file

from src.election.election_authority import ElectionAuthority
from src.election.trustee.trustee_client import init_trustees
from src.election.bulletin_board.bulletin_board_client import BulletinBoardClient
from src.election.election_properties import ElectionProperties
from src.election.borda.borda_election_system import Borda

from src.protocols.protocol_suite import ProtocolSuite
from src.protocols.sublinear import SubLinearProtocolSuite 

from src.crypto.abb import ABB
from src.crypto.paillier_abb import PaillierABB

from src.util.utils import get_class_from_name, subclasses_recursive

log = logging.getLogger(__name__)


class Authentication:
    def __init__(self):
        self.bulletin_board_client = None
        self.election_id = None
        self.election_authority = None

    def add_ballot(self, vote):
        """
        Called when a vote is received from the frontend.
        This function can perform certain prechecks, like whether the vote was
        received during the election timeframe. Furthermore it can add some
        additional metadata like the date the vote was received.

        Args:
            vote (dict): The vote as it was received from the frontend.

        Returns:
            tuple: (errorcode, ack_code, times_voted, timestamp)
        """
        config: ElectionProperties = self.bulletin_board_client.get_election_config()

        # Add timestamp to votes
        timestamp = datetime.now(timezone.utc)
        vote["bullet"]["timestamp"] = timestamp.isoformat()

        # Test if the vote is received during the election duration
        if not (config.start_date <= timestamp <= config.due_date):
            return -10, None, 0, None

        # Check if User is allowed to vote, i.e. has a hash that is listed on the bulletin board
        secret_hash = hashlib.sha512(base64.b16decode(vote["secret"]))
        hash_hex = secret_hash.hexdigest()

        log.info("User with secret %.8s... and hash %.8s... tries to vote", vote["secret"], hash_hex)

        if not self.bulletin_board_client.is_hash_valid(hash_hex):
            return -20, None, 0, None

        times_voted = self.timesVoted(hash_hex)

        # Send Vote
        vote["bullet"]["hash"] = hash_hex
        self.bulletin_board_client.add_vote(vote["bullet"])

        # generate ack
        ack = self.generateACK(vote["voterId"], vote["electionId"])
        log.info("Successful vote, sent ACK '%s' to client.", ack)
        return 0, ack, times_voted, timestamp

    def timesVoted(self, hash_hex):
        ballots = self.bulletin_board_client.get_votes()
        count = 1  # add current vote
        for ballot in ballots:
            if ballot["hash"] == hash_hex:
                count += 1
        return count

    def generateACK(self, voter_id, election_id):
        return "ACK_E" + str(election_id) + "V" + str(voter_id)

    def createElection(self, raw_config):
        """ raw_config is the config json sent by the frontend. it looks like the following:
            {
                candidates: Array or an integer (array prefered) ,
                ballot: {
                    category: String //from this a ballot format is selected,
                    settings: {
                        minNumberOfChoices: int,
                        maxNumberOfChoices: int,
                        numberOfEqualRanks: int,
                        numberOfCountingRanks: int,
                        pointDistribution: [] of ints,
                        points: int,
                        allowDraw: bool,
                        spreadPointsAccross: bool,
                    } // Settings the frontend needs to render a ballot
                },
                election: {
                    id: int,
                    title: String,
                    start: String of datetime in ISOformat,
                    due: String of datetime in ISOformat,
                    numberOfVoters: int,
                },
                communication: {
                    peerServerIP: String of hostname, current default:"localhost",
                    authenticationServerIP: String of IP, current default: "https://localhost:9001/",
                    bulletinBoardServerIP: String of IP, current default: "https://localhost:9002/",
                },
                encryption: {
                    ABB: String of ABB classname e.g. "PaillierABB",
                    protocolSuite:  String of ProtocolSuite classname e.g. "SubLinearProtocolSuite",
                    bits: int power of 2 - save above 2048,
                    numberShares: int,
                    threshold: int,
                    publicKey: String generated from ciphertext,
                },
                evaluation: {
                    type: String of ElectionProperties classname e.g. "Borda",
                    //TODO Fill Settings and display them in the form
                    settings: {} // Settings the trustees need to evaluate an election
                }
            },
        """
        log.debug("Received config from frontend: %s", raw_config)
        log.debug("The prescribed bulletin board server is: '%s'", raw_config["communication"]["bulletinBoardServerIP"])

        # Generate the abbs which are needed by trustees
        abb_cls: ABB = get_class_from_name(raw_config["encryption"]["ABB"], subclasses_recursive(ABB))
        prot_suite_cls = get_class_from_name(raw_config["encryption"]["protocolSuite"], subclasses_recursive(ProtocolSuite))
        abbs = abb_cls.gen_trustee_abbs(
            bits=raw_config["encryption"]["bits"],
            num_shares=raw_config["encryption"]["numberShares"],
            threshold=raw_config["encryption"]["threshold"],
            prot_suite_cls=prot_suite_cls,
        )
        def key_generator(bulletin_board): return init_trustees(bulletin_board, abbs, [i for i, _ in enumerate(abbs)])

        # Create an ElectionProperties object with the configs taken form the frontend
        election_properties = ElectionProperties.deserialize(raw_config)

        # Create the corresponding election authority
        self.election_authority = ElectionAuthority(key_generator, election_properties)
        self.election_id = self.election_authority.bulletin_board.board_id
        self.bulletin_board_client = BulletinBoardClient(board_id=self.election_id)
        self.bulletin_board_client.session = self.election_authority.bulletin_board.session

        # generate Ballots
        self.election_authority.generate_ballots(raw_config["election"]["numberOfVoters"],
                                                 raw_config["communication"]["bulletinBoardServerIP"])

    def trigger_evaluation(self):
        return self.election_authority.trigger_evaluation()
