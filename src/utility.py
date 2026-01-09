from datetime import datetime, timedelta, timezone

def next_forex_trading_day(feature_date_utc: datetime) -> datetime.date:
    """
    Returns the next valid Forex trading date (UTC-based).
    """
    weekday = feature_date_utc.weekday() # Monday=0, Sunday=6

    # Friday
    if weekday == 4:
        return (feature_date_utc + timedelta(days=3)).date()

    # Saturday
    if weekday == 5:
        return (feature_date_utc + timedelta(days=2)).date()

    # Sunday
    if weekday == 6:
        # Before market open (22:00 UTC) → Monday
        if feature_date_utc.hour < 22:
            return (feature_date_utc + timedelta(days=1)).date()
        else:
            return feature_date_utc.date()

    # Monday–Thursday
    return (feature_date_utc + timedelta(days=1)).date()
    

