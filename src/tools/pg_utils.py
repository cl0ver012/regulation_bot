import json
import os
import asyncpg
import numpy as np
from pinecone import Pinecone
from dotenv import load_dotenv
import asyncio
import time

# Load environment variables from .env file
load_dotenv()

from openai import OpenAI
client = OpenAI()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., "us-west1-gcp"
INDEX_NAME = os.getenv("INDEX_NAME")
pinecone_vector_str = os.getenv("PINECONE_VECTOR")
pinecone_vector_array = json.loads(pinecone_vector_str)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai.api_key = OPENAI_API_KEY

# PostgreSQL connection parameters
RDS_HOST = "database-test1.cz044iyc6ix3.us-east-1.rds.amazonaws.com"
RDS_PORT = int(os.getenv("RDS_PORT", 5432))  # Default to 5432 if not set
RDS_DATABASE = os.getenv("RDS_DATABASE")
RDS_USER = os.getenv("RDS_USER")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)


def fetch_data_from_pinecone():
    """Fetch all vectors and their metadata from the Pinecone index."""
    try:
        # Check if the index exists
        if INDEX_NAME not in pc.list_indexes().names():
            print(f"Index {INDEX_NAME} does not exist.")
            return []

        index = pc.Index(INDEX_NAME)

        # Initialize an empty list to hold records
        records = []

        # Get all vector IDs and their data
        cursor = None
        while True:
            response = index.query(
                vector=pinecone_vector_array,
                namespace="legislacao",
                top_k=553,
                include_values=True,
                include_metadata=True,
                cursor=cursor,
            )

            for vector in response["matches"]:
                records.append(
                    {
                        "id": vector["id"],  # Get the ID of the vector
                        "alinea": vector.get("Metadata", {}).get("alinea"),
                        "artigo": vector.get("Metadata", {}).get("artigo"),
                        "capitulo": vector.get("Metadata", {}).get("capitulo"),
                        "inciso": vector.get("Metadata", {}).get("inciso"),
                        "paragrafo": vector.get("Metadata", {}).get("paragrafo"),
                        "pos": vector.get("Metadata", {}).get("pos"),
                        "secao": vector.get("Metadata", {}).get("secao"),
                        "subsecao": vector.get("Metadata", {}).get("subsecao"),
                        "texto": vector.get("Metadata", {}).get("texto"),
                        "titulo": vector.get("Metadata", {}).get("titulo"),
                        "query": vector["values"],  # Get the vector data
                    }
                )

            # Check if there are more results to fetch
            if "next_cursor" not in response:
                break
            cursor = response["next_cursor"]

        return records

    except Exception as e:
        print(f"Error fetching data from Pinecone: {e}")
        return []


def list_to_postgres_array(list_data):
    """Convert a list to PostgreSQL array string format."""
    return "{" + ",".join(map(str, list_data)) + "}"


def list_to_json_array(list_data):
    return json.dumps(list_data)


async def insert_data_to_postgres(records):
    """Insert records into PostgreSQL database with exponential backoff."""
    retries = 5
    for attempt in range(retries):
        conn = None
        try:
            print("------------Started---------------")
            # Establish a connection to the PostgreSQL database
            conn = await asyncpg.connect(
                host=RDS_HOST,
                port=RDS_PORT,
                database=RDS_DATABASE,
                user=RDS_USER,
                password=RDS_PASSWORD,
            )

            print("-----------DB connected---------------")

            async with conn.transaction():  # Create a transaction context
                # Prepare data for insertion
                formatted_data = [
                    (
                        record["alinea"],
                        record["artigo"],
                        record["capitulo"],
                        record["inciso"],
                        record["paragrafo"],
                        record["pos"],
                        record["secao"],
                        record["subsecao"],
                        record["texto"],
                        record["titulo"],
                        list_to_json_array(
                            record["query"]
                        ),  # Convert list to JSON array string
                        record["id"],
                    )
                    for record in records
                ]

                # Modify the insert query to include new fields
                insert_query = """
                INSERT INTO legislacao (
                    alinea, artigo, capitulo, inciso, paragrafo,
                    pos, secao, subsecao, texto, titulo, query,
                    pinecone_id
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12);
                """

                # Use executemany to batch insert records
                await conn.executemany(insert_query, formatted_data)

            # Commit the changes automatically when the transaction completes
            print(f"Inserted {len(formatted_data)} records into PostgreSQL.")
            return  # Exit function on successful insertion

        except Exception as e:
            print(f"Error inserting data into PostgreSQL: {e}")
            # Exponential backoff strategy
            wait_time = 2**attempt
            print(f"Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
        finally:
            # Ensure the connection is closed
            if conn:
                await conn.close()
    print("Failed to insert data into PostgreSQL after multiple attempts.")


async def generate_embedding(sentence):
    try:
        response = client.embeddings.create(
            input="Your text string goes here",
            model="text-embedding-3-small"
        )

        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    


async def query_postgres_with_pgvector(query_sentence):
    conn = None
    try:
        query_embedding = await generate_embedding(query_sentence)
        if query_embedding is None:
            return

        conn = await asyncpg.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            database=RDS_DATABASE,
            user=RDS_USER,
            password=RDS_PASSWORD,
        )

        query_embedding_str = list_to_json_array(query_embedding)

        query = """
            SELECT *, 1 - (query <=> $1) AS cosine_similarity
            FROM legislacao
            ORDER BY cosine_similarity DESC
            LIMIT 10;
        """

        rows = await conn.fetch(query, query_embedding_str)
        
        result = []

        for row in rows:
            print(
                f"Pinecone ID: {row['pinecone_id']}, Similarity: {row['cosine_similarity']}, Texto: {row['texto']}"
            )
            result.append(row['texto'])
        
        return result

    except Exception as e:
        print(f"Error querying PostgreSQL with pgvector: {e}")
    finally:
        if conn:
            await conn.close()
