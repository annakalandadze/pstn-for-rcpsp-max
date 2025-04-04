from typing import List

from rcpsp_max.entities.successors import Successor


class Job:
    def __init__(self, id, duration, successors: List[Successor], resources, std=0):
        self.id = id
        self.duration = duration
        self.successors = successors
        self.resources = resources
        self.std = std
        self.normalized_duration = duration
        self.normalized_std = std

    def to_dict(self):
        return {
            "id": self.id,
            "duration": self.duration,
            "successors": [succ.to_dict() for succ in self.successors],
            "resources": self.resources
        }

    @staticmethod
    def from_dict(data):
        return Job(
            id=data["id"],
            duration=data["duration"],
            successors=[Successor.from_dict(succ) for succ in data["successors"]],
            resources=data["resources"]
        )