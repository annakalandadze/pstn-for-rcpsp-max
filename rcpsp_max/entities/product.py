from typing import List

from rcpsp_max.entities.job import Job


class Product:
    def __init__(self, id, jobs: List[Job], demand, inventory, makespan=0):
        self.id = id
        self.jobs = jobs
        self.demand = demand
        self.inventory = inventory
        self.min_makespan = makespan
        self.value = 0

    def to_dict(self):
        return {
            "id": self.id,
            "jobs": [job.to_dict() for job in self.jobs],  # Serialize jobs
            "demand": self.demand,
            "inventory": self.inventory,
            "min_makespan": self.min_makespan
        }

    @classmethod
    def from_dict(cls, data):
        jobs = [Job.from_dict(job) for job in data["jobs"]]
        return cls(data["id"], jobs, data["demand"], data["inventory"], data["min_makespan"])
