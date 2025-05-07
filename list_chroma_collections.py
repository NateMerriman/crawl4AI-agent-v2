#!/usr/bin/env python3
"""
list_chroma_collections.py
--------------------------
Command-line utility to connect to a ChromaDB database and list all collections
along with the number of items (documents) each collection contains.

Usage:
    python list_chroma_collections.py --db-dir <path_to_db_directory>
    python list_chroma_collections.py --db-dir ./chroma_db
"""

import argparse
import sys
import os

try:
    import chromadb
except ImportError:
    print(
        "ChromaDB library not found. Please install it by running: pip install chromadb"
    )
    sys.exit(1)


def list_collections_with_counts(db_dir: str):
    """Connects to ChromaDB and lists all collections with their item counts."""

    print(f"Connecting to ChromaDB at directory: {db_dir}")
    try:
        client = chromadb.PersistentClient(path=db_dir)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        print(
            f"Please ensure the directory 	'{db_dir}	' exists and is a valid ChromaDB database path."
        )
        sys.exit(1)

    print("Fetching list of collections...")
    try:
        collections = client.list_collections()
    except Exception as e:
        print(f"Error listing collections: {e}")
        sys.exit(1)

    if not collections:
        print("No collections found in this ChromaDB instance.")
        return

    print(f"Found {len(collections)} collection(s):")
    print("--------------------------------------------------")
    print(f"{'Collection Name':<40} | {'Item Count':<10}")
    print("--------------------------------------------------")

    for collection_obj in collections:
        collection_name = collection_obj.name
        try:
            # Get the collection to count items
            collection_to_count = client.get_collection(name=collection_name)
            count = collection_to_count.count()
            print(f"{collection_name:<40} | {count:<10}")
        except Exception as e:
            print(
                f"Could not retrieve or count items for collection 	'{collection_name}	': {e}"
            )
            print(f"{collection_name:<40} | {'Error counting':<10}")
    print("--------------------------------------------------")


def main():
    parser = argparse.ArgumentParser(
        description="List all collections in a ChromaDB instance with item counts."
    )
    parser.add_argument(
        "--db-dir",
        required=True,
        help="Directory path of the ChromaDB database (e.g., ./chroma_db)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        print(
            f"Error: The ChromaDB directory 	'{args.db_dir}	' does not exist or is not a directory."
        )
        sys.exit(1)

    list_collections_with_counts(args.db_dir)


if __name__ == "__main__":
    main()
