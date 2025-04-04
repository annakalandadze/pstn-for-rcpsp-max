class Successor:
    def __init__(self, id, lag):
        self.id = id
        self.lag = lag

    def to_dict(self):
        return {
            "id": self.id,
            "lag": self.lag
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["lag"])