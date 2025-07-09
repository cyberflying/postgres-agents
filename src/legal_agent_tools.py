import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from datetime import datetime
import json
from typing import Any, Callable, Set
from sqlalchemy import create_engine
from azure.ai.agents.telemetry import trace_function
from opentelemetry import trace
from urllib.parse import quote_plus


# Load environment variables
load_dotenv("../.env")
CONN_STR = os.getenv("AZURE_PG_CONNECTION")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")

def dsn_to_uri(dsn: str) -> str:
    """
    Convert psycopg2-style DSN to SQLAlchemy URI.
    Example DSN:
        host=psql.example.com dbname=postgres user=admintest password=Pa55 port=5432 sslmode=require
    """
    parts = dict(p.split("=", 1) for p in dsn.split())
    return (
        f"postgresql+psycopg2://"
        f"{quote_plus(parts['user'])}:{quote_plus(parts['password'])}@"
        f"{parts['host']}:{parts.get('port', '5432')}/"
        f"{parts.get('dbname', parts.get('database', 'postgres'))}"
        f"?sslmode={parts.get('sslmode', 'require')}"
    )
SQLA_URI = CONN_STR if "://" in CONN_STR else dsn_to_uri(CONN_STR)


# The trace_func decorator will trace the function call and enable adding additional attributes
# to the span in the function implementation. Note that this will trace the function parameters and their values.

# Get data from the Postgres database
@trace_function()
def vector_search_cases(vector_search_query: str, start_date: datetime ="1911-01-01", end_date: datetime ="2025-12-31", limit: int = 10) -> str:
    """
    Fetches the cases information in Washington State for the specified query.

    :param query(str): The query to fetch cases for specifically in Washington.
    :type query: str
    :param start_date: The start date for the search, defaults to "1911-01-01"
    :type start_date: datetime, optional
    :param end_date: The end date for the search, defaults to "2025-12-31"
    :type end_date: datetime, optional
    :param limit: The maximum number of cases to fetch, defaults to 10
    :type limit: int, optional

    :return: Cases information as a JSON string.
    :rtype: str
    """
        
    db = create_engine(SQLA_URI)
    
    query = """
    SELECT id, name, opinion,
        opinions_vector <=> azure_openai.create_embeddings(
            %s,
            %s)::vector 
        AS similarity
    FROM cases
    WHERE decision_date BETWEEN %s AND %s
    ORDER BY similarity
    LIMIT %s;
    """
    
    # Fetch cases information from the database
    df = pd.read_sql(
        query,
        db,
        params=(
            embedding_model_name,
            vector_search_query,
            datetime.strptime(start_date, "%Y-%m-%d"),
            datetime.strptime(end_date, "%Y-%m-%d"),
            limit,
        ),
    )

    # Adding attributes to the current span
    span = trace.get_current_span()
    span.set_attribute("requested_query", query)

    cases_json = json.dumps(df.to_json(orient="records"))
    span.set_attribute("cases_json", cases_json)
    return cases_json

# Count the number of cases related to the specified query
@trace_function()
def count_cases(vector_search_query: str, limit: int = 10) -> str:
    """
    Count the number of cases related to the specified query.
    Invoke this tool for aggregation, if the user mentions the word "count" or "how many" or a similar word.

    :param query(str): The query to search.
    :type query: str
    :param limit: The maximum number of cases fetch, defaults to 10
    :type limit: int, optional

    :return: listing information as a JSON string.
    :rtype: str
    """

    db = create_engine(SQLA_URI)
    
    query = """
    SELECT COUNT(*)
    FROM cases
    WHERE opinions_vector <=> azure_openai.create_embeddings(
            %s,
            %s)::vector < 0.8 
    LIMIT %s;
    """

    df = pd.read_sql(query, db, params=(embedding_model_name, vector_search_query, limit))

    # Adding attributes to the current span
    span = trace.get_current_span()
    span.set_attribute("requested_query", query)
    cases_count = json.dumps(df.to_json(orient="records"))
    span.set_attribute("result", cases_count)

    return cases_count

# Statically defined user functions for fast reference
user_functions: Set[Callable[..., Any]] = {
    vector_search_cases,
    count_cases
}
