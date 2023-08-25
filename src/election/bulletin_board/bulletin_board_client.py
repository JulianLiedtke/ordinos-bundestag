import json
import requests
from src.election.election_properties import ElectionProperties
from src.election.bulletin_board.bulletin_board import BulletinBoard
from src.network.bearer_auth import HTTPBearerAuth

class BulletinBoardClient(BulletinBoard):

    base_url = 'https://localhost:9002'
    verify_path = 'certificate_localhost/cert.pem'

    def __init__(self, board_id=None):
        self.session = requests.Session()
        self.session.verify = self.verify_path
        self.session.headers.update({'content-type': 'application/json'})

        if board_id is None:
            self.__add_bb()
        else:
            self.board_id = board_id

    def __add_bb(self):
        url = self.base_url + '/api/addBulletinBoard'
        response = self.session.post(url=url)
        if response.ok:
            response_json = response.json()
            self.board_id = response_json["board_id"]
            self.session.auth = HTTPBearerAuth(response_json["token"])
        return self.board_id

    def add_vote(self, vote):
        url = self.base_url + '/api/addVote'
        payload = {
            "vote": vote,
            "board_id": self.board_id
        }
        response = self.session.post(url=url, json=payload)
        if response.ok:
            pass

    def get_votes(self):
        url = self.base_url + '/api/getVotes'
        payload = {
            "board_id": self.board_id
        }
        response = self.session.get(url=url, params=payload)
        if response.ok:
            return response.json()["votes"]

    def set_election_config(self, config):
        url = self.base_url + '/api/setConfig'
        payload = {
            "config": config,
            "board_id": self.board_id
        }
        response = self.session.post(url=url, json=payload)
        if response.ok:
            pass

    def get_election_config(self):
        url = self.base_url + '/api/getConfig'
        payload = {
            "board_id": self.board_id
        }
        response = self.session.get(url=url, params=payload)
        response.raise_for_status()
        properties = ElectionProperties.deserialize(response.json())
        return properties

    def add_hash(self, h):
        url = self.base_url + '/api/addHash'
        payload = {
            "hash": h,
            "board_id": self.board_id
        }
        response = self.session.post(url=url, json=payload)
        if response.ok:
            pass

    def is_hash_valid(self, h):
        url = self.base_url + '/api/validHash'
        payload = {
            "hash": h,
            "board_id": self.board_id
        }
        response = self.session.post(url=url, json=payload)
        if response.ok:
            print("response is good")
            is_valid = response.json()["is_valid"]
            return is_valid

    def add_trustee_message(self, from_id, message):
        url = self.base_url + '/api/addMessage'
        payload = {
            "board_id": self.board_id,
            "from_id": from_id,
            "message": message
        }
        response = self.session.post(url=url, json=payload)
        response.raise_for_status()

    def get_trustee_messages(self, from_id, last_index):
        url = self.base_url + '/api/getMessages'
        payload = {
            "board_id": self.board_id,
            "from_id": from_id,
            "last_index": last_index
        }
        response = self.session.get(url=url, params=payload)
        response.raise_for_status()
        return response.json()["messages"]

    def add_result(self, from_id, result):
        url = self.base_url + '/api/addResult'
        payload = {
            "board_id": self.board_id,
            "from_id": from_id,
            "result": result
        }
        response = self.session.post(url=url, json=payload)
        response.raise_for_status()

    def get_results(self):
        url = self.base_url + '/api/getResults'
        payload = {
            "board_id": self.board_id
        }
        response = self.session.get(url=url, params=payload)
        if response.ok:
            return response.json()["results"]
