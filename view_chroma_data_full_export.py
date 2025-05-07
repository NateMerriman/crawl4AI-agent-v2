#!/usr/bin/env python3
"""
view_chroma_data_full_export.py
-------------------------------
Command-line utility to connect to a ChromaDB database, fetch all documents (chunks)
and their metadata from a specified collection, and export them to a CSV file.

Usage:
    python view_chroma_data_full_export.py --collection <collection_name> --db-dir <path_to_db_directory> --output-csv <path_for_output.csv>
    python view_chroma_data_full_export.py --db-dir ./chroma_db --collection n8n --output-csv ./n8n_data_export.csv
"""

import argparse
import csv
import sys
import os
import traceback  # For detailed error reporting

try:
    import chromadb

    try:
        CHROMA_VERSION = chromadb.__version__
    except AttributeError:
        CHROMA_VERSION = "unknown (chromadb module has no __version__ attribute)"
    print(f"DEBUG: Using chromadb version: {CHROMA_VERSION}")
except ImportError:
    print(
        "ChromaDB library not found. Please install it by running: pip install chromadb"
    )
    sys.exit(1)


def export_collection_to_csv(
    db_dir: str, target_collection_name: str, output_csv_file: str
):
    """Connects to ChromaDB, fetches data from the collection, and writes it to a CSV file."""

    print(
        f"DEBUG: export_collection_to_csv received target_collection_name: {repr(target_collection_name)}"
    )
    print(f"Connecting to ChromaDB at directory: {db_dir}")

    try:
        client = chromadb.PersistentClient(path=db_dir)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        traceback.print_exc()
        sys.exit(1)

    collection_to_export = None
    try:
        print(
            f"Listing all collections to find an exact match for {repr(target_collection_name)}..."
        )
        all_collections = client.list_collections()
        if not all_collections:
            print("No collections found in the database.")
            sys.exit(1)

        print("Available collections found (iterating with repr to see exact names):")
        for coll_obj in all_collections:
            print(f"  - Iterating. Name: {repr(coll_obj.name)}, ID: {coll_obj.id}")
            if coll_obj.name == target_collection_name:
                print(
                    f"    DEBUG: Exact match found! Comparing {repr(coll_obj.name)} == {repr(target_collection_name)}"
                )
                collection_to_export = coll_obj
                break

        if collection_to_export is None:
            print(
                f"Error: Collection with the exact name {repr(target_collection_name)} was not found."
            )
            sys.exit(1)

        collection = collection_to_export
        print(
            f"DEBUG: Using matched collection object: name={repr(collection.name)}, id={collection.id}"
        )
        item_count = collection.count()
        print(
            f"Successfully targeted collection {repr(collection.name)}. It reports {item_count} items."
        )

    except Exception as e:
        print(
            f"An error occurred while trying to find or access collection {repr(target_collection_name)}:"
        )
        print(f"  Error type: {type(e)}")
        print(f"  Error repr: {repr(e)}")
        traceback.print_exc()
        sys.exit(1)

    print(
        f"DEBUG: About to call .get() on collection with name: {repr(collection.name)} and id: {collection.id} to fetch ALL items."
    )

    results = None
    ids = []
    documents = []
    metadatas = []

    try:
        # Fetch ALL items from the collection
        results = collection.get(include=["metadatas", "documents"])
        retrieved_ids_count = len(results.get("ids", []))
        print(
            f"DEBUG: Results from .get() (all items): retrieved {retrieved_ids_count} items."
        )
        if retrieved_ids_count > 0:
            print(f"DEBUG: First retrieved ID (of all): {results['ids'][0]}")
            # print(f"DEBUG: First retrieved document (first 50 chars): {results["documents"][0][:50]}") # Can be verbose
            # print(f"DEBUG: First retrieved metadata: {results["metadatas"][0]}") # Can be verbose

    except Exception as e:
        print(f"Error during collection.get() for {repr(collection.name)}:")
        print(f"  Error type: {type(e)}")
        print(f"  Error repr: {repr(e)}")
        traceback.print_exc()
        print(
            "Continuing to attempt CSV export with any data retrieved before the error, if any."
        )
        ids = results.get("ids", []) if isinstance(results, dict) else []
        documents = results.get("documents", []) if isinstance(results, dict) else []
        metadatas = results.get("metadatas", []) if isinstance(results, dict) else []

    if not ids and isinstance(results, dict):
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

    if not ids:
        print(
            f"No items actually retrieved or an error prevented retrieval from collection {repr(collection.name)}. The CSV file will be empty or not created."
        )
        if not os.path.exists(output_csv_file) or os.path.getsize(output_csv_file) == 0:
            header_for_empty = [
                "id",
                "document",
                "source",
                "chunk_index",
                "headers",
                "char_count",
                "word_count",
            ]
            try:
                with open(
                    output_csv_file, "w", newline="", encoding="utf-8"
                ) as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=header_for_empty)
                    writer.writeheader()
                print(f"Created an empty CSV with headers: {output_csv_file}")
            except Exception as e_csv:
                print(f"Could not create empty CSV: {e_csv}")
        return

    print(f"Found {len(ids)} items to write. Preparing CSV: {output_csv_file}")

    header = ["id", "document"]
    if metadatas:
        all_meta_keys = set()
        for meta_item in metadatas:
            if meta_item:
                all_meta_keys.update(meta_item.keys())
        sorted_meta_keys = sorted(list(all_meta_keys))
        header.extend(sorted_meta_keys)

    try:
        with open(output_csv_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for i in range(len(ids)):
                row = {
                    "id": ids[i],
                    "document": documents[i]
                    if documents and i < len(documents)
                    else None,
                }
                meta_for_row = metadatas[i] if metadatas and i < len(metadatas) else {}
                if meta_for_row:
                    for key in sorted_meta_keys:
                        row[key] = meta_for_row.get(key)
                writer.writerow(row)
        print(f"Successfully wrote {len(ids)} items to {output_csv_file}")
    except IOError as e:
        print(f"Error writing to CSV file {output_csv_file}: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Export ChromaDB collection to a CSV file with debug output."
    )
    parser.add_argument(
        "--db-dir",
        required=True,
        help="Directory path of the ChromaDB database (e.g., ./chroma_db)",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Name of the collection to export (e.g., docs)",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the output CSV file (e.g., ./chroma_export.csv)",
    )

    args = parser.parse_args()

    print(f"DEBUG: Raw args.collection from command line: {repr(args.collection)}")
    collection_name_from_arg = args.collection.strip()
    print(
        f"DEBUG: Stripped collection_name_from_arg for script use: {repr(collection_name_from_arg)}"
    )

    if not os.path.isdir(args.db_dir):
        print(
            f"Error: The ChromaDB directory {args.db_dir} does not exist or is not a directory."
        )
        sys.exit(1)

    export_collection_to_csv(args.db_dir, collection_name_from_arg, args.output_csv)


if __name__ == "__main__":
    main()
