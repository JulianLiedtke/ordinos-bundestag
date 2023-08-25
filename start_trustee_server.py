"""
Create a new server which will run in a separte thread to represent one Trustee
"""
import logging
from time import sleep
from src.election.trustee.trustee_server import TrusteeWebserver
from src.util.logging import setup_logging

if __name__ == '__main__':
    setup_logging(logging.INFO)
    servers = []
    for port in range(9003, 9005):
        webserver = TrusteeWebserver(port=port, separate_thread=True)
        servers.append(webserver)
    try:
        while True:
            sleep(0.5)
    except KeyboardInterrupt:
        pass
