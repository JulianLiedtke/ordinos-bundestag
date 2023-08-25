from http import HTTPStatus
import json
import logging
from functools import partial
from src.network.bulletin_board_connector import BulletinBoardChannel
from src.election.bulletin_board.bulletin_board_client import BulletinBoardClient
from typing import Any, Dict
from src.network.webserver import CustomBaseHTTPRequestHandler, HTTPSServer
from src.protocols.protocol import Protocol
from src.crypto.paillier_abb import PaillierABB
from src.election.trustee.trustee import Trustee

# The following modules need to be loaded, so that they may be deserialized.
# So that we don't create the possibility to load arbitrary modules at runtime
# we load them beforehand during the init of the module.

# Import the protocol suites that may be used, so that they can be deserialized
from src.protocols.sublinear import SubLinearProtocolSuite

# Import the election systems that may be used, so that they can be deserialized
from src.election.borda import borda_election_system, borda_implementations
from src.election.condorcet import condorcet_election_system, condorcet_implementations
from src.election.single_vote.single_vote_election_system import SingleVoteElection
from src.election.parliamentary_ballot.parliamentary_evaluation import ParliamentaryEvaluation
from src.election.real_world_elections.bundestagelection import BundestagElection

# Import the evaluator functions that may be used, so that they can be deserialized
import src.election.evaluation
from src.election.condorcet import condorcet_evaluation, condorcet_no_winner_evaluations


log = logging.getLogger(__name__)


class TrusteeWebserver(HTTPSServer):

    def __init__(self, host='', port=9003, separate_thread=False):
        """
        Create a new trustee webserver which may host multiple trustees at once.

        Args:
            host (str, optional): The host address of the server. Defaults to '' which binds to all interfaces.
            port (int, optional): The port on which the server is available. Defaults to 9002.
            separate_thread (bool, optional): Should the server run in a separate thread. Defaults to False.
        """
        self.trustees: Dict[Any, Trustee] = {}
        super().__init__(partial(_TrusteeHTTPRequestHandler, self), host, port, separate_thread)

    def init_new_trustee(self, board_id, abb, broadcasting_trustee_id, other_trustee_ids):
        """
        Adds a new trustee to the list of trustees, with a newly generated id.

        Returns:
            int: The newly created id under which one can find the trustee.
        """
        new_id = self._generate_new_trustee_id()
        bb_client = BulletinBoardClient(board_id=board_id)
        bb_channel = BulletinBoardChannel(bb_client, broadcasting_trustee_id, other_trustee_ids)
        self.trustees[new_id] = Trustee(abb, new_id, bb_client, bb_channel)
        return new_id

    def _generate_new_trustee_id(self):
        """
        Generate a new trustee id which is not yet represented in the list of trustees.

        Returns:
            int: A yet unused id for the next trustee
        """
        return str(len(self.trustees.keys()))


class _TrusteeHTTPRequestHandler(CustomBaseHTTPRequestHandler):
    """
    This Class implements the request handling of the TrusteeWebserver.
    """
    def log_message(self, format, *args):
        return

    def get_function_callback(self):
        get_function_callbacks = {
            "/status/": self.api_get_status,
            "/protocol/": self.api_get_protocol
        }
        post_function_callbacks = {
            "/": self.api_add_trustee,
            "/protocol/": self.api_run_protocol,
            "/evaluation/": self.api_trigger_evaluation
        }

        try:
            if self.command == "GET":
                path = self.path.split('?', 1)[0]
                return get_function_callbacks[path]
            elif self.command == "POST":
                return post_function_callbacks[self.path]
        except KeyError:
            pass
        raise NotImplementedError

    def api_add_trustee(self, board_id, abb, from_id, to_ids):
        # TODO Generalize ABB deserialization
        new_id = self.host_server.init_new_trustee(board_id, PaillierABB.deserialize(abb), from_id, to_ids)
        response = {"trustee_id": new_id}
        log.info("Trustee [%s]: Was added for BB '%s'", (self.host_server.port, new_id), board_id)
        response = json.dumps(response).encode("utf-8")
        self.write_headers_and_response(HTTPStatus.CREATED, response)

    def api_run_protocol(self, trustee_id, protocol, protocol_args):
        acting_trustee = self.host_server.trustees[trustee_id]
        # TODO Deserialize protocol properly 
        acting_trustee.run_protocol(Protocol.deserialize(protocol), protocol_args)
        self.write_headers_and_response(HTTPStatus.OK)

    def api_get_status(self, trustee_id):
        acting_trustee = self.host_server.trustees[trustee_id]
        response = {"is_finished": acting_trustee.is_protocol_finished()}
        if acting_trustee.is_protocol_finished():
            response["result"] = acting_trustee.result
        response = json.dumps(response).encode("utf-8")
        self.write_headers_and_response(HTTPStatus.OK, response)

    def api_get_protocol(self, trustee_id):
        acting_trustee = self.host_server.trustees[trustee_id]
        response = {"protocol": acting_trustee.get_protocol()}
        response = json.dumps(response).encode("utf-8")
        self.write_headers_and_response(HTTPStatus.OK, response)

    def api_trigger_evaluation(self, trustee_id):
        log.info("Trustee [%s]: %s", (self.host_server.port, trustee_id), "starts evaluation")
        acting_trustee = self.host_server.trustees[trustee_id]
        acting_trustee.trigger_evaluation()
        self.write_headers_and_response(HTTPStatus.OK)
