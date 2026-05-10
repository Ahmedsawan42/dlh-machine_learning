#!/usr/bin/env python3
"""
Function to list all documents in a MongoDB collection
"""


def list_all(mongo_collection):
    """
    List all documents in a collection

    Args:
        mongo_collection: pymongo collection object

    Returns:
        List of documents (empty list if no documents)
    """
    documents = []
    for doc in mongo_collection.find():
        documents.append(doc)
    return documents
