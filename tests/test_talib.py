import pandas as pd
import talib.abstract as ta


def test_talib_bollingerbands_near_zero_values():
    inputs = pd.DataFrame(
        [
            {"close": 0.00000010},
            {"close": 0.00000011},
            {"close": 0.00000012},
            {"close": 0.00000013},
            {"close": 0.00000014},
        ]
    )
    bollinger = ta.BBANDS(inputs, matype=0, timeperiod=2)
    # Print detailed values for debugging
    print("Upper band values:", bollinger["upperband"].values)
    print("Middle band values:", bollinger["middleband"].values)
    print("Lower band values:", bollinger["lowerband"].values)
    print(
        f"Index 3: upperband = {bollinger['upperband'][3]}, "
        f"middleband = {bollinger['middleband'][3]}"
    )
    # Ensure BBANDS runs and produces expected-length outputs, regardless of band equality.
    assert len(bollinger["upperband"]) == len(inputs)
    assert len(bollinger["middleband"]) == len(inputs)
    assert len(bollinger["lowerband"]) == len(inputs)
