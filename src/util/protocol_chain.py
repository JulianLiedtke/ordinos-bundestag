from src.protocols.protocol import Protocol


class ChainedProtocol(Protocol):

    def __init__(self, protocols):
        super().__init__()
        self.protocols = protocols

    def run(self, *initial_args):
        last_result = initial_args
        for protocol in self.protocols:
            last_result = self.run_subprotocol(protocol, last_result)
        return last_result