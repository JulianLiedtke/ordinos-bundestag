import logging
from src.network.bulletin_board_connector import BulletinBoardChannel
from src.election.bulletin_board.bulletin_board_client import BulletinBoardClient
import requests
from src.election.trustee.trustee import Trustee

log = logging.getLogger(__name__)


class TrusteeClient(Trustee):

    verify_path = 'certificate_localhost/cert.pem'

    def __init__(self, abb, id, board_id, channel, base_url="https://localhost:9003"):
        self.session = requests.Session()
        self.session.verify = self.verify_path
        self.session.headers.update({'content-type': 'application/json'})

        self.base_url = base_url
        self.board_id = board_id
        self.channel = channel
        self.abb = abb
        self.id_on_bulletin_board = id
        self.id_on_trustee_server = self.add_on_trustee_server()

    def add_on_trustee_server(self):
        url = self.base_url + '/'
        log.info("Adding trustee on server '%s' with abb '%s'", self.base_url, self.abb.serialize())
        payload = {"board_id": self.board_id,
                   "abb": self.abb.serialize(),
                   "from_id": self.channel.id,
                   "to_ids": self.channel.other_trustee_ids}
        response = self.session.post(url=url, json=payload)
        response.raise_for_status()
        return response.json()["trustee_id"]

    def run_protocol(self, prot, *args):
        # TODO Add proper (de-)serialization, only proof of concept currently
        url = self.base_url + '/protocol/'
        payload = {"trustee_id": self.id_on_trustee_server,
                   "protocol": prot.serialize(),
                   "protocol_args": prot.serialize_run_args(args)}
        response = self.session.post(url=url, json=payload)
        response.raise_for_status()

    def get_protocol(self):
        # TODO Add proper (de-)serialization, only proof of concept currently
        url = self.base_url + '/protocol/'
        payload = {"trustee_id": self.id_on_trustee_server}
        response = self.session.get(url=url, params=payload)
        response.raise_for_status()
        return response.json()["protocol"]

    def is_protocol_finished(self):
        url = self.base_url + '/status/'
        payload = {"trustee_id": self.id_on_trustee_server}
        response = self.session.get(url=url, params=payload)
        response.raise_for_status()
        return response.json()["is_finished"]

    @property
    def result(self):
        url = self.base_url + '/status/'
        payload = {"trustee_id": self.id_on_trustee_server}
        response = self.session.get(url=url, params=payload)
        response.raise_for_status()
        return response.json().get("result", None)

    def trigger_evaluation(self):
        url = self.base_url + '/evaluation/'
        payload = {"trustee_id": self.id_on_trustee_server}
        response = self.session.post(url=url, json=payload)
        response.raise_for_status()


def init_trustees(bulletin_board, abbs, ids):

    if not isinstance(bulletin_board, BulletinBoardClient):
        raise TypeError("Remote trustees (TrusteeClient) only support remote bulletin boards (BulletinBoardClient)")

    trustees = []
    n_trustees = len(ids)
    trustee_server_addresses = ["https://localhost:{}".format(9003 + i) for i in range(n_trustees)]

    log.debug("Creating trustees with ids : '%s'", ids)

    # create trustees
    for i in range(n_trustees):
        other_ids = ids.copy()
        other_ids.pop(i)
        channel = BulletinBoardChannel(bulletin_board, ids[i], other_ids)
        trustee = TrusteeClient(abbs[i], ids[i], bulletin_board.board_id, channel, trustee_server_addresses[i])
        trustees.append(trustee)

    return trustees
