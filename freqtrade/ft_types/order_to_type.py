from dataclasses import dataclass
from typing import Literal

from freqtrade.constants import LongShort


# Used for Orders preparation for execution on exchange or backtesting
@dataclass
class OrderToCreate:
    pair: str
    type: Literal["limit", "market"]
    side: Literal["buy", "sell"]
    price: float
    trigger_price: float | None
    amount: float
    stake_amount: float
    leverage: float
    order_tag: str
    reduce_only: bool
    time_in_force: Literal["GTC", "FOK", "IOC"]
    trade_side: LongShort


@dataclass
class OrderToValidate:
    type: Literal["limit", "market"]
    side: Literal["buy", "sell"]
    price: float
    trigger_price: float | None
    amount: float
    action_side: Literal["entry", "exit"]
    time_in_force: Literal["GTC", "FOK", "IOC"]
    order_tag: str
    trade_side: LongShort


def convert_to_validated_orders(custom_orders: list[dict]) -> list[OrderToValidate]:
    """
    Converts a list of dictionaries to a list of OrderToValidate instances.
    Validates that all required fields are present (except optional ones like trigger_price)
    and checks for unexpected fields.

    Args:
        custom_orders (list[dict]): The list of order dictionaries to convert.

    Returns:
        list[OrderToValidate]: The validated and converted orders.

    Raises:
        ValueError: If a field is missing, has an invalid type, or there are unexpected fields.
    """
    required_fields = {
        "type": str,
        "side": str,
        "price": (int, float),
        "amount": (int, float),
        "action_side": str,
        "time_in_force": str,
        "order_tag": str,
        "trade_side": str,
    }
    optional_fields = {
        "trigger_price": (int, float, type(None)),
    }
    validated_orders = []

    for order_dict in custom_orders:
        if not isinstance(order_dict, dict):
            raise ValueError(
                f"Each order must be a dictionary, got {type(order_dict)}: {order_dict}"
            )

        # Check for missing fields (required only)
        missing_fields = [key for key in required_fields if key not in order_dict]
        if missing_fields:
            raise ValueError(
                f"Order is missing required fields: {missing_fields}. Order: {order_dict}"
            )

        # Check for unexpected fields
        all_expected_fields = set(required_fields) | set(optional_fields)
        unexpected_fields = [key for key in order_dict if key not in all_expected_fields]
        if unexpected_fields:
            raise ValueError(
                f"Order has unexpected fields: {unexpected_fields}. Order: {order_dict}"
            )

        # Validate field types for required fields
        for field, expected_type in required_fields.items():
            if not isinstance(order_dict[field], expected_type):
                raise ValueError(
                    f"Field '{field}' has invalid type. Expected {expected_type}, "
                    f"got {type(order_dict[field])}. Value: {order_dict[field]}"
                )

        # Validate field types for optional fields if present
        for field, expected_type in optional_fields.items():
            if field in order_dict and not isinstance(order_dict[field], expected_type):
                raise ValueError(
                    f"Field '{field}' has invalid type. Expected {expected_type}, "
                    f"got {type(order_dict[field])}. Value: {order_dict[field]}"
                )

        # Convert to OrderToValidate instance
        validated_order = OrderToValidate(
            type=order_dict["type"],
            side=order_dict["side"],
            price=order_dict["price"],
            trigger_price=order_dict.get("trigger_price"),  # Use .get() for optional field
            amount=order_dict["amount"],
            action_side=order_dict["action_side"],
            time_in_force=order_dict["time_in_force"],
            order_tag=order_dict["order_tag"],
            trade_side=order_dict["trade_side"],
        )
        validated_orders.append(validated_order)

    return validated_orders
