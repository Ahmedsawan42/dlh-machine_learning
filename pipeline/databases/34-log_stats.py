#!/usr/bin/env python3
"""
Script to provide statistics about Nginx logs stored in MongoDB
"""

from pymongo import MongoClient


def log_stats():
    """Display statistics about Nginx logs"""
    client = MongoClient("mongodb://127.0.0.1:27017")
    collection = client.logs.nginx
    
    # Total number of logs
    total_logs = collection.count_documents({})
    print("{} logs".format(total_logs))
    
    # Methods statistics
    print("Methods:")
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))
    
    # Status check statistics
    status_check = collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print(f"{status_check} status check")

    client.close()

if __name__ == "__main__":
    log_stats()
