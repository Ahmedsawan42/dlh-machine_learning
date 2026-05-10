#!/usr/bin/env python3
"""
Function to return schools with a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Returns the list of schools having a specific topic

    Args:
        mongo_collection: pymongo collection object
        topic: topic to search for (string)

    Returns:
        List of school documents that have the specified topic
    """
    return list(mongo_collection.find({"topics": topic}))
