from abc import abstractmethod
import functools
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import logging
import ssl
import socket
import threading
import json
from urllib.parse import parse_qsl

log = logging.getLogger(__name__)


class HTTPSServer:
    """
    This class creates acts as a generic HTTPS server. Starts the web server upon creation.
    """
    certfile = 'certificate_localhost/cert.pem'
    keyfile = 'certificate_localhost/key.pem'
    certpath = 'certificate_localhost/'

    def __init__(self, RequestHandler, host, port, separate_thread=False):
        """
        Start the HTTPS server. Utilizes the RequestHandler to parse and process all requests.
        The request handler also obtains a reference to the server in its first argument
        so that its state may be changed.

        Args:
            RequestHandler (RequestHandler): The handler class which will process the requests.
            host (str, optional): The host name of the server.
            port (int, optional): The port of the server.
            separate_thread (bool, optional): Should the server run in a separate thread (True)
                                              or in the calling thread (False). Defaults to False.
        """
        self.host = host
        self.port = port

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_verify_locations(capath=self.certpath)
        context.verify_mode = ssl.CERT_NONE
        context.hostname_checks_common_name = False
        context.check_hostname = False
        context.load_cert_chain(self.certfile, self.keyfile)

        self.server = ThreadingHTTPServer((self.host, self.port), RequestHandler)

        self.server.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.socket.bind((self.host, self.port))
        self.server.socket.listen(5)
        self.server.socket = context.wrap_socket(self.server.socket, server_side=True)

        log.info("Starting HTTPS %s on https://%s:%s/", self.__class__.__name__, *((self.host, self.port) if self.host else self.server.socket.getsockname()))

        if separate_thread:
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
        else:
            self.server.serve_forever()


def require_auth(func):
    """
    Function decorator that makes sure a valid auth is given for a request handled by the request 
    handlerbefore the function is called. To check whether the given auth is valid it requires the
     auth_check method of the request handler class to be overwritten.
    """
    @functools.wraps(func)
    def wrapper_check_auth(self, *args, **kwargs):
        if self.auth_check(*args, **kwargs):
            return func(self, *args, **kwargs)
        self.write_headers_and_response(HTTPStatus.UNAUTHORIZED,
                                        additional_headers=[("WWW-Authenticate", "Bearer")])
    return wrapper_check_auth


class CustomBaseHTTPRequestHandler(BaseHTTPRequestHandler):

    def __init__(self, host_server: HTTPSServer, *args, **kwargs):
        """
        Create a new handler for an incoming message to the host server

        Args:
            host_server (HTTPSServer): A reference to the calling server.
        """
        self.host_server = host_server
        self.protocol_version = 'HTTP/1.1'
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self):
        """
        Handle the OPTIONS request which responds with the allowed communication capabilities
        of the server.
        """
        self.send_response(HTTPStatus.OK)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With")
        self.send_header("Connection", "keep-alive")
        self.send_header("keep-alive", "timeout=5")
        self.send_header("Content-Length", 0)
        self.end_headers()

    def do_GET(self):
        """
        Pass on all incoming GET requests to their corresponding sub-handler function.

        Raises:
            exc: Re-raised from the sub-handler, should he raise one.
        """
        try:
            arguments = {}
            if '?' in self.path:
                _, query = self.path.split('?', 1)
                arguments = dict(parse_qsl(query))
            self.get_function_callback()(**arguments)
        except NotImplementedError:
            self.write_headers_and_response(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.write_headers_and_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            raise exc

    def do_POST(self):
        """
        Pass on all incoming POST requests to their corresponding sub-handler function.

        Raises:
            exc: Re-raised from the sub-handler, should he raise one.
        """
        try:
            content_length = int(self.headers['Content-Length'])
            arguments = {}

            if content_length > 0:
                post_data = self.rfile.read(content_length).decode()
                arguments = json.loads(post_data)
                
            self.get_function_callback()(**arguments)
        except NotImplementedError as exc:
            self.write_headers_and_response(HTTPStatus.NOT_IMPLEMENTED)
        except Exception as exc:
            self.write_headers_and_response(HTTPStatus.INTERNAL_SERVER_ERROR)
            raise exc

    @abstractmethod
    def get_function_callback(self):
        raise NotImplementedError()

    @abstractmethod
    def auth_check(self):
        return True

    def write_headers_and_response(self, response_code, response=None, keep_alive=True, additional_headers=None):
        self.send_response(response_code)
        self.send_header("Connection", "keep-alive" if keep_alive else "close")
        if keep_alive:
            self.send_header("keep-alive", "timeout=5")

        if additional_headers:
            for (keyword, value) in additional_headers:
                self.send_header(keyword, value)

        if response is None:
            self.send_header('Content-Length', 0)
            self.end_headers()
        else:
            self.send_header('Content-Length', len(response))
            self.send_header('Content-Type', 'application/json;chareset=utf-8')
            self.end_headers()
            self.wfile.write(response)
