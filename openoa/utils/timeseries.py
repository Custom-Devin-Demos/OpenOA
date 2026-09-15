"""
This module provides useful functions for processing timeseries data
"""

from __future__ import annotations

import datetime
from typing import Any, TypeVar, Callable, cast

import numpy as np
import pandas as pd
from pytz import utc, timezone
from dateutil.parser import parse

from openoa.utils._converters import series_method

_F = TypeVar("_F", bound=Callable[..., Any])


def _series_method(data_cols: list[str]) -> Callable[[_F], _F]:
    """Signature-preserving view of :py:func:`openoa.utils._converters.series_method`, which is
    not yet typed; the wrapper is a pass-through to the decorated function's arguments."""
    return cast("Callable[[_F], _F]", series_method(data_cols=data_cols))


def _as_series(col: pd.Series | str) -> pd.Series:
    """Narrows a ``series_method`` argument to the pandas ``Series`` the wrapper guarantees."""
    if isinstance(col, str):
        raise TypeError(f"Column name '{col}' was provided without the `data` argument.")
    return col


def offset_to_seconds(offset: int | float | str | np.timedelta64) -> int | float:
    """Converts pandas datetime offset alias to its corresponding number of seconds.

    Args:
        offset(:obj:`int` | :obj:`float` | :obj:`str` | :obj:`numpy.datetime64`): The pandas offset
            alias or numpy timestamp to be converted to seconds. If a number (int or
            float) is passed, then it must be in nanoseconds, the Pandas default.

    Returns:
        :obj:`int` | `float`: The number of seconds corresponding to :py:attr:`offset`.
    """
    value: int | float | str | pd.Timedelta = (
        pd.Timedelta(offset) if isinstance(offset, np.timedelta64) else offset
    )
    try:
        seconds = pd.to_timedelta(value).total_seconds()
    except ValueError:  # Needs a leading number or the above will fail
        seconds = pd.to_timedelta(f"1{value}").total_seconds()
    return seconds


def determine_frequency_seconds(data: pd.DataFrame, index_col: str | None = None) -> int | float:
    """Calculates the most common time difference between all non-duplicate timestamps and returns
    that difference in seconds.

    Args:
        data(:obj:`pandas.DataFrame`): The pandas DataFrame to determine the DatetimeIndex frequency.
        index_col(:obj:`str` | `None`, optional): The name of the index column if :py:attr:`data`
            uses a MultiIndex, otherwise leave as None. Defaults to None.

    Returns:
        :obj:`int` | `float`: The number of seconds corresponding to :py:attr:`offset`.
    """
    # Get the non-duplicated DatetimeIndex values from a single level, or multi-level index
    index = data.index if index_col is None else data.index.get_level_values(index_col)
    index = index.unique()

    unique_diffs, counts = np.unique(np.diff(index.to_numpy()), return_counts=True)
    return offset_to_seconds(unique_diffs[np.argmax(counts)])


def determine_frequency(data: pd.DataFrame, index_col: str | None = None) -> str | int | float:
    """Gets the offset alias from the datetime index of :py:attr:`data`, or calculates the most
    common time difference between all non-duplicate timestamps.

    Args:
        data(:obj:`pandas.DataFrame`): The pandas DataFrame to determine the DatetimeIndex frequency.
        index_col(:obj:`str` | `None`, optional): The name of the index column if :py:attr:`data`
            uses a MultiIndex, otherwise leave as None. Defaults to None.

    Returns:
        :obj:`str` | :obj:`int` | :obj:`float`: The offset string or number of seconds between timestamps.
    """
    # Get the timetamp index values
    index = data.index if index_col is None else data.index.get_level_values(index_col)
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("The index of `data` (or its `index_col` level) must be a DatetimeIndex.")

    # Check for an offset string being available
    freq: str | None = index.freqstr
    if freq is None:
        time_index = data.index.get_level_values("time")
        if not isinstance(time_index, pd.DatetimeIndex):
            raise TypeError("The 'time' index level of `data` must be a DatetimeIndex.")
        freq = pd.infer_freq(time_index)

    # If there is at least one missing data point, or timestamp misalignment, the above will fail,
    # so
    if freq is None:
        return determine_frequency_seconds(data, index_col)
    return freq


def convert_local_to_utc(d: str | datetime.datetime, tz_string: str) -> datetime.datetime:
    """
    Convert timestamps in local time to UTC. The function can only act on a single timestamp at a time, so
    for example use the .apply function in Pandas:

        date_utc = df['time'].apply(convert_local_to_utc, args = ('US/Pacific',))

    Also note that this function doesn't solve the end of DST when times between 1:00-2:00 are repeated
    in November. Those dates are left repeated in UTC time and need to be shifted manually.

    The function does address the missing 2:00-3:00 times at the start of DST in March

    Args:
        d(:obj:`datetime.datetime`): the local date, tzinfo must not be set
        tz_string(:obj:`str`): the local timezone

    Returns:
        :obj:`datetime.datetime`: the local date converted to UTC time

    """
    # TODO: Make a second copy of this method that aligns with the QA.convert_datetime_column method
    if isinstance(d, str):
        d = parse(d)
    if not isinstance(d, datetime.datetime):
        raise TypeError(
            "The input to `d` must be a `datetime.datetime` object or a string that can be converted to one."
        )

    # Define the timezone, and convert to a the localized timestamp as needed
    tz = timezone(tz_string)
    # TODO: Figure out why a datetime object with tzinfo encoded is different than localizing with pytz
    d_local = d if d.tzinfo else tz.localize(d, is_dst=True)
    return d_local.astimezone(utc)  # calculate UTC time


@_series_method(data_cols=["dt_col"])
def convert_dt_to_utc(
    dt_col: pd.Series | str, tz_string: str, data: pd.DataFrame | None = None
) -> pd.Series:
    """Converts a pandas ``Series`` of timestamps, string-formatted or ``datetime.datetime`` objects
        that are in a local timezone ``tz_string`` to a UTC encoded pandas ``Series``.

    Args:
        dt_col (:obj:`pandas.Series` | `str`): A pandas ``Series`` of datetime objects or
            string-encoded timestamps, or a the name of the column in `data`.
        tz_string (str): The string name for the expected timezone of the provided timestamps in :py:attr:`dt_col`.
        data (:obj:`pandas.DataFrame`, optional): The pandas ``DataFrame`` containing the timestamp
            column: :py:attr:`dt_col`. Defaults to None.

    Returns:
        pd.Series: _description_
    """
    series = _as_series(dt_col)
    if isinstance(series[0], str):
        series = series.apply(parse)

    # If the timezone information is already encoded, then convert it to a UTC-converted
    # pandas datetime object automatically, otherwise, localize it, then convert it
    first: datetime.datetime = series[0]
    if first.tzinfo is not None:
        converted: pd.Series = pd.to_datetime(series, utc=True)
        return converted
    localized: pd.Series = series.dt.tz_localize(tz_string, ambiguous=True).dt.tz_convert(utc)
    return localized


@_series_method(data_cols=["dt_col"])
def find_time_gaps(
    dt_col: pd.Series | pd.DatetimeIndex | str, freq: str, data: pd.DataFrame | None = None
) -> pd.Series:
    """
    Finds gaps in `dt_col` based on the expected frequency, `freq`, and returns them.

    Args:
        dt_col(:obj:`pandas.Series`): Pandas ``Series`` of ``datetime.datetime`` objects or the name
            of the column in :py:attr:`data`.
        freq(:obj:`string`): The expected frequency of the timestamps, which should align with
            the pandas timestamp conventions (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases).
        data (:obj:`pandas.DataFrame`, optional): The pandas ``DataFrame`` containing the timestamp
            column: :py:attr:`dt_col`. Defaults to None.

    Returns:
        :obj:`pandas.Series`: Series of missing time stamps in ``datetime.datetime`` format
    """
    if isinstance(dt_col, pd.DatetimeIndex):
        dt_col = dt_col.to_series()
    dt_col = _as_series(dt_col)

    # If the difference for all of the timestamps is the expected frequency, 0 (duplicate), or a NaT
    # (first element of `diff`), then return an empty series
    if np.all(dt_col.diff().isin([pd.Timedelta(freq), pd.Timedelta("0"), pd.NaT])):
        return pd.Series([], name=dt_col.name, dtype="object")

    # Create a date range object and return a Series of the set difference of both objects
    range_dt = pd.Series(data=pd.date_range(dt_col.min(), end=dt_col.max(), freq=freq))
    return pd.Series(tuple(set(range_dt).difference(dt_col)), name=dt_col.name)


@_series_method(data_cols=["dt_col"])
def find_duplicate_times(
    dt_col: pd.Series | pd.DatetimeIndex | str, data: pd.DataFrame | None = None
) -> pd.Series:
    """
    Find duplicate input data and report them. The first duplicated item is not reported, only subsequent duplicates.

    Args:
        dt_col(:obj:`pandas.Series` | `str`): Pandas series of ``datetime.datetime`` objects or the name of the
            column in :py:attr:`data`.
        data (:obj:`pandas.DataFrame`, optional): The pandas `DataFrame` containing the timestamp
            column: :py:attr:`dt_col`. Defaults to None.

    Returns:
        :obj:`pandas.Series`: Duplicates from input data
    """
    if isinstance(dt_col, pd.DatetimeIndex):
        dt_col = dt_col.to_series()
    dt_col = _as_series(dt_col)

    return dt_col[dt_col.duplicated()]


def gap_fill_data_frame(data: pd.DataFrame, dt_col: str, freq: str) -> pd.DataFrame:
    """
    Insert any missing timestamps into :py:attr:`data` while filling the data columns with NaNs.

    Args:
        data(:obj:`pandas.DataFrame`): The dataframe with potentially missing timestamps.
        dt_col(:obj:`str`): Name of the column in 'data' with timestamps.
        freq(:obj:`str`): The expected frequency of the timestamps.

    Returns:
        :obj:`pandas.DataFrame`: output data frame with NaN data for the data gaps

    """
    # If the dataframe is empty, just return it.
    if data.empty:
        return data

    missing_dt = find_time_gaps(data[dt_col], freq)
    if (not missing_dt.empty) and missing_dt.notnull().any():
        gap_df = pd.DataFrame(columns=data.columns)
        gap_df[dt_col] = missing_dt
        if not (gap_df.empty or gap_df.isna().values.sum() == gap_df.size):
            data = pd.concat([data, gap_df], axis=0)
    try:
        return data.sort_values(dt_col)
    except ValueError:
        # Catches when dt_col and the index name are the same, and temporarily renames the column
        # to perform the sort, and puts it back for the returned data
        return (
            data.rename(columns={dt_col: f"_{dt_col}"})
            .sort_values(f"_{dt_col}")
            .rename(columns={f"_{dt_col}": dt_col})
        )


@_series_method(data_cols=["col"])
def percent_nan(col: pd.Series | str, data: pd.DataFrame | None = None) -> float:
    """
    Return percentage of data that are Nan or 1 if the series is empty.

    Args:
        col(:obj:`pandas.Series`): The pandas `Series` to be checked for NaNs, or the name of the
            column in :py:attr:`data`.
        data (:obj:`pandas.DataFrame`, optional): The pandas ``DataFrame`` containing the timestamp
            column: :py:attr:`col`. Defaults to None.

    Returns:
        :obj:`float`: Percentage of NaN data in the data series
    """
    col = _as_series(col)
    if (denominator := float(col.size)) == 0:
        return 1
    return int(np.isnan(col.to_numpy()).sum()) / denominator


@_series_method(data_cols=["dt_col"])
def num_days(dt_col: pd.Series | str, data: pd.DataFrame | None = None) -> int:
    """
    Calculates the number of non-duplicate days in :py:attr:`dt_col`.

    Args:
        dt_col(:obj:`pandas.Series` | str): A pandas ``Series`` with a timeseries index to be checked
            for the number of days contained in the data.
        data (:obj:`pandas.DataFrame`, optional): The pandas ``DataFrame`` containing the timestamp
            column: :py:attr:`dt_col` and having a timeseries index. Defaults to None.

    Returns:
        :obj:`int`: Number of days in the data
    """
    dt_col = _as_series(dt_col)
    return dt_col[~dt_col.index.duplicated()].resample("D").asfreq().index.size


@_series_method(data_cols=["dt_col"])
def num_hours(dt_col: pd.Series | str, *, data: pd.DataFrame | None = None) -> int:
    """
    Calculates the number of non-duplicate hours in `dt_col`.

    Args:
        dt_col(:obj:`pandas.Series` | str): A pandas ``Series`` of timeseries data to be checked for
            the number of hours contained in the data
        data (:obj:`pandas.DataFrame`, optional): The pandas `DataFrame` containing the timestamp
            column: :py:attr:`dt_col`. Defaults to None.

    Returns:
        :obj:`int`: Number of hours in the data
    """
    dt_col = _as_series(dt_col)
    return dt_col[~dt_col.index.duplicated()].resample("h").asfreq().index.size
