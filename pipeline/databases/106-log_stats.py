#!/usr/bin/env python3
"""
Script to provide statistics about Nginx logs stored in MongoDB
with top 10 most present IPs
"""

from pymongo import MongoClient


def log_stats():
    """
    Display statistics about Nginx logs from MongoDB
    """
    # Connect to MongoDB - use the accepted format
    client = MongoClient("mongodb://127.0.0.1:27017")
    collection = client.logs.nginx

    # Get total number of logs
    total_logs = collection.count_documents({})
    print("{} logs".format(total_logs))

    # Methods to count
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    # Print Methods header
    print("Methods:")

    # Count and print each method
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    # Count documents with method=GET and path=/status
    status_count = collection.count_documents({
        "method": "GET",
        "path": "/status"
    })
    print("{} status check".format(status_count))

    # Get top 10 most present IPs
    print("IPs:")

    pipeline = [
        {
            "$group": {
                "_id": "$ip",
                "count": {"$sum": 1}
            }
        },
        {
            "$sort": {"count": -1}
        },
        {
            "$limit": 10
        }
    ]

    top_ips = list(collection.aggregate(pipeline))

    for ip_data in top_ips:
        print("\t{}: {}".format(ip_data['_id'], ip_data['count']))

    # Close connection
    client.close()


if __name__ == "__main__":
    log_stats()