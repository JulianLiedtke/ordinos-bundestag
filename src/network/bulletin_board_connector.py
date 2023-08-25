from src.network.connection import Channel, Connector
from time import sleep
import logging
import requests

log = logging.getLogger(__name__)


class BulletinBoardConnector(Connector):

    def __init__(self, bulletin_board, sender_id):
        self.id = sender_id
        self.bulletin_board = bulletin_board
        self.receive_buffer = []
        self.last_received_index = 0

    def connect(self, connection_id):
        self.connection_id = connection_id

    def send(self, msg):
        pass

    def receive(self):
        wait_time = 0
        while len(self.receive_buffer) == 0:
            new_messages = self.bulletin_board.get_trustee_messages(self.connection_id, self.last_received_index)
            self.receive_buffer.extend(new_messages)
            self.last_received_index += len(new_messages)
            if len(self.receive_buffer) != 0:
                break
            # Here we busy wait for new data to appear
            # Make sending while waiting for new data possible through threading or something similar
            # TODO replace with a callback from bb server, which it will call as soon as new data is accessible
            # sleep(wait_time)
            # wait_time = wait_time * 2 + 0.1
        return self.receive_buffer.pop(0)


class BulletinBoardChannel(Channel):
    def __init__(self, bulletin_board, broadcasting_trustee_id, other_trustee_ids):
        self.bulletin_board = bulletin_board
        self.other_trustee_ids = [str(other) for other in other_trustee_ids]
        self.id = str(broadcasting_trustee_id)
        self.connectors = self.init_bb_connectors()

    def init_bb_connectors(self):
        connectors = []
        for other_id in self.other_trustee_ids:
            connector = BulletinBoardConnector(self.bulletin_board, self.id)
            connector.connect(other_id)
            connectors.append(connector)
        return connectors

    def broadcast(self, msg):
        self.bulletin_board.add_trustee_message(str(self.id), msg)
