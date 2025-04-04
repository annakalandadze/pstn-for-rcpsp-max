from typing import List

from rcpsp_max.entities.product import Product


class Order:
    def __init__(self, id, products: List[Product], deadline, required, value):
        self.id = id
        self.products = products
        self.deadline = deadline
        self.required = required
        self.value = value

    def to_dict(self):
        return {
            "id": self.id,
            "products": [product.to_dict() for product in self.products],  # Serialize products
            "deadline": self.deadline,
            "required": self.required,
            "value": self.value
        }

    @classmethod
    def from_dict(cls, data):
        products = [Product.from_dict(prod) for prod in data["products"]]
        return cls(data["id"], products, data["deadline"], data["required"], data["value"])