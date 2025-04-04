class Resource:
    def __init__(self, id, capacity):
        self.id = id
        self.capacity = capacity

    def to_dict(self):
        return {
            "id": self.id,
            "capacity": self.capacity
        }

    @classmethod
    def from_dict(cls, data):
        return cls(data["id"], data["capacity"])
