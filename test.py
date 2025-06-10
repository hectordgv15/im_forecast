import pytest
import pandas as pd
import numpy as np

from IMForecast.ts_functions import ts_functions

@pytest.fixture
def fc():
    return ts_functions()


def test_error_on_overlapping_dates(fc):
    orig = pd.Series(
        [1, 2, 3],
        index = pd.date_range('2025-03-31', periods = 3, freq = 'Q')
    )
    diff = pd.Series(
        [1],
        index = pd.date_range('2025-09-30', periods = 1, freq = 'Q')
    )
    with pytest.raises(ValueError) as exc:
        fc.invert_diff_forecast(diff, orig, d = 1, D = 0, m = 4)
    assert "Última fecha de original_series" in str(exc.value)


def test_no_differences_returns_same_series(fc):
    original = pd.Series(
        [5, 6, 7],
        index=pd.date_range('2025-03-31', periods = 3, freq = 'Q')
    )
    forecast_diff = pd.Series(
        [10, 20],
        index = pd.date_range('2025-12-31', periods = 2, freq = 'Q')
    )
    result = fc.invert_diff_forecast(forecast_diff, original, d = 0, D = 0, m = 4)
    result = result.astype(original.dtype)
    
    pd.testing.assert_series_equal(result, forecast_diff)


def test_regular_differencing_inversion(fc):
    original = pd.Series(
        [1, 2, 5],
        index = pd.date_range('2025-03-31', periods = 3, freq = 'Q')
    )
    forecast_diff = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        index = pd.date_range('2025-12-31', periods = 14, freq = 'Q')
    )
    
    expected = pd.Series(
        [6, 8, 11, 15, 20, 26, 33, 41, 50, 60, 71, 83, 96, 110],
        index = forecast_diff.index
    )
    
    result = fc.invert_diff_forecast(forecast_diff, original, d = 1, D = 0, m = 4)
    result = result.astype(original.dtype)
    pd.testing.assert_series_equal(result, expected)


def test_seasonal_differencing_inversion(fc):
    original = pd.Series(
        [0, 1, 2, 3, 4],
        index=pd.date_range('2023-12-31', periods = 5, freq = 'Q')
    )

    # Diferencias estacionales pronosticadas: seis trimestres de 2025–2026
    forecast_diff = pd.Series(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        index = pd.date_range('2025-03-31', periods = 11, freq = 'Q')
    )
    
    expected = pd.Series(
        [2, 4, 6, 8, 7, 10, 13, 16, 16, 20, 24],
        index = forecast_diff.index
    )
    
    result = fc.invert_diff_forecast(forecast_diff, original, d = 0, D = 1, m = 4)
    result = result.astype(original.dtype)
    pd.testing.assert_series_equal(result, expected)


def test_regular_and_seasonal_differencing_inversion(fc):
    original = pd.Series(
        [12.69, 12.73, 12.99, 13.38, 11.67],
        index = pd.date_range('2022-06-30', periods = 5, freq = 'Q')
    )
    
    forecast_diff = pd.Series(
        [0.18, -0.35, 0.1, 0.69, -0.28, -0.51, 0.26],
        index = pd.date_range('2023-09-30', periods = 7, freq = 'Q')
    )
    expected = pd.Series(
        [11.89, 11.8, 12.29, 11.27, 11.21, 10.61, 11.36],
        index = forecast_diff.index
    )
    result = fc.invert_diff_forecast(forecast_diff, original, d = 1, D = 1, m = 4)
    result = result.astype(original.dtype)
    pd.testing.assert_series_equal(result, expected)
    
    