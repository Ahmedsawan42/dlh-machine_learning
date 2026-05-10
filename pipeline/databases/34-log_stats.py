#!/usr/bin/env python3
"""
Script to provide statistics about Nginx logs stored in MongoDB
"""

from pymongo import MongoClient


def log_stats():
    """Display statistics about Nginx logs"""
    client = MongoClient('mongodb://localhost:27017/')
    db = client.logs
    collection = db.nginx

    # Total number of logs
    total_logs = collection.count_documents({})
    print(f"{total_logs} logs")

    # Methods statistics
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    for method in methods:
        count = collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    # Status check statistics
    status_check = collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print(f"{status_check} status check")

    client.close()


if __name__ == "__main__":
    log_stats()
