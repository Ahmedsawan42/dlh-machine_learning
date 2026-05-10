#!/usr/bin/env python3
"""
Function to update topics for a school document
"""


def update_topics(mongo_collection, name, topics):
    """
    Changes all topics of a school document based on the name

    Args:
        mongo_collection: pymongo collection object
        name: school name to update (string)
        topics: list of strings representing topics approached in the school

    Returns:
        None (updates the document in place)
    """
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
