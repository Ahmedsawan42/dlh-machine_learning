#!/usr/bin/env python3
"""
Function to insert a new document into a MongoDB collection
"""


def insert_school(mongo_collection, **kwargs):
    """
    Insert a new document into a collection based on kwargs

    Args:
        mongo_collection: pymongo collection object
        **kwargs: key-value pairs representing the document fields

    Returns:
        The new _id of the inserted document
    """
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
