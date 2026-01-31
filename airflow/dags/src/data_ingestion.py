from twelvedata import TDClient
import os
from dotenv import load_dotenv

load_dotenv()

class TwelveDataIngestor():
    def __init__(self, api_key=None):
        api_key = api_key or os.environ.get("TWELEVEDATA_API_KEY")
        if not api_key:
            raise ValueError("TWELEVEDATA_API_KEY not provided")
        self.client = TDClient(apikey=api_key)

    def ingest(self, symbol):
        ts = self.client.time_series(
            symbol=symbol,
            interval="1day",
            outputsize=1,
            timezone="UTC"
        )
        data = ts.as_json()
        print(f"Symbol {symbol} data successfully retrieved.")
        return data


if __name__ == "__main__":
    data_ingestor = TwelveDataIngestor()
    data = data_ingestor.ingest(symbol="EUR/USD")

    print(data)