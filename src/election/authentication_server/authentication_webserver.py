from http import HTTPStatus
from src.network.webserver import CustomBaseHTTPRequestHandler, HTTPSServer
from urllib.parse import parse_qsl
from functools import partial

from src.election.authentication_server.authentication import Authentication

import json
import logging

log = logging.getLogger(__name__)


class AuthenticationWebserver(HTTPSServer):
    """
    This class creates a HTTP server for the authentication webserver. Upon creation it starts
    the web server.
    """

    certfile = 'certificate_localhost/cert.pem'
    keyfile = 'certificate_localhost/key.pem'
    certpath = 'certificate_localhost/'

    def __init__(self, host='', port=9001, separate_thread=False):
        """
        Sets up a new authentication webserver which hosts the auth to a bulletin board.
        All client requests which add to the bulletin board have to go through this server beforehand.
        Currently also acts as a ElectionAuthority intermediary server.

        Args:
            host (str, optional): The host name of the server. Defaults to '' which binds to all interfaces.
            port (int, optional): The port of the server. Defaults to 9001.
            separate_thread (bool, optional): Should this sever run in a new thread. Defaults to False.
        """
        self.authentication = Authentication()
        super().__init__(partial(_AuthenticationHTTPRequestHandler, self), host, port, separate_thread=separate_thread)


class _AuthenticationHTTPRequestHandler(CustomBaseHTTPRequestHandler):
    """
    This Class implements the request handling of the AuthenticationWebserver.
    """

    def get_function_callback(self):
        get_function_callbacks = {
        }
        post_function_callbacks = {
            "/api/addBallot": self.api_add_ballot,
            "/api/createElection": self.api_create_election,
            "/api/triggerEvaluation": self.api_trigger_evaluation
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

    def api_add_ballot(self, ballot):
        errorcode, ack, times_voted, timestamp = self.host_server.authentication.add_ballot(ballot)
        response = {
            "ack": ack,
            "timesVoted": times_voted,
            "errorcode": errorcode,
            "timestamp": timestamp.isoformat() if timestamp is not None else None
        }
        response = json.dumps(response).encode("utf-8")
        self.write_headers_and_response(HTTPStatus.CREATED, response=response, additional_headers=[('Access-Control-Allow-Origin', '*')])

    def api_create_election(self, raw_config):
        self.host_server.authentication.createElection(raw_config)
        self.write_headers_and_response(HTTPStatus.CREATED, additional_headers=[('Access-Control-Allow-Origin', '*')])

    def api_trigger_evaluation(self):
        self.host_server.authentication.trigger_evaluation()
        self.write_headers_and_response(HTTPStatus.OK, additional_headers=[('Access-Control-Allow-Origin', '*')])
