"""Tests for Supabase database boundary handling."""

from types import SimpleNamespace

from src.data.database import SupabaseClient


class _FakeSupabaseTable:
    def __init__(self, data):
        self.data = data

    def select(self, _columns):
        return self

    def execute(self):
        return SimpleNamespace(data=self.data)


class _FakeSupabaseClient:
    def __init__(self, data):
        self.data = data

    def table(self, _table):
        return _FakeSupabaseTable(self.data)


def test_get_table_data_empty_date_filtered_response_returns_empty_frame():
    client = object.__new__(SupabaseClient)
    client.client = _FakeSupabaseClient([])

    df = client._get_table_data(
        "posts",
        start_date="2023-01-01",
        end_date="2023-01-31",
    )

    assert df.empty
