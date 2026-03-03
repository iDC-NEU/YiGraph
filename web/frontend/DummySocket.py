import json

class DummySocket:
        async def send(self, msg):
            print("[TEST OUTPUT]", msg)
            self.returnmsg = msg

        def __init__(self,msg11):
            self.message = msg11
