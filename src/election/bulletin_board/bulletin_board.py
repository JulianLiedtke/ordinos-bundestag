import logging
from time import time

log = logging.getLogger(__name__)

# TODO: Make the data stored by the board persistent across sessions (i.e. save to a file and load from it)
class BulletinBoard:

    def __init__(self):
        self.votes = []
        self.hashes = []
        self.election_config = None
        self.trustee_messages = {}
        self.results = {}

    def add_vote(self, vote):
        """
        add vote to list of votes
        | vote has to be valid
        """
        self.votes.append(vote)

    def add_hash(self, h):
        self.hashes.append(h)

    def set_election_config(self, serialized_config):
        self.election_config = serialized_config

    def get_election_config(self):
        return self.election_config

    def get_votes(self):
        """returns list of votes"""
        return self.votes

    def add_trustee_message(self, from_id, message):
        self.trustee_messages.setdefault(from_id, []).append(message)

    def get_trustee_messages(self, from_id, last_index):
        messages = self.trustee_messages.get(from_id, [])
        return messages[int(last_index):]

    def is_hash_valid(self, h):
        return h in self.hashes

    def add_result(self, from_id, result):
        self.results[from_id] = result

    def get_results(self):
        return self.results
