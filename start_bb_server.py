from time import sleep
from src.election.bulletin_board.bulletin_board_webserver import BulletinBoardWebserver
from src.util.logging import setup_logging
import logging

"""
Create a new server which will run in a separte thread to represent bulletin boards
"""
if __name__ == '__main__':
    setup_logging(logging.INFO)
    try:
        s = BulletinBoardWebserver()
    except KeyboardInterrupt:
        pass
