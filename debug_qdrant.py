
try:
    from qdrant_client import QdrantClient
    print(f"QdrantClient imported successfully.")
    print(f"Attributes of QdrantClient: {dir(QdrantClient)}")
    
    # Try to instantiate and check instance attributes if possible without connecting
    # client = QdrantClient(":memory:")
    # print(f"Instance attributes: {dir(client)}")
    
    if hasattr(QdrantClient, 'search'):
        print("search method FOUND in QdrantClient class")
    else:
        print("search method NOT FOUND in QdrantClient class")

    import qdrant_client
    print(f"Version: {qdrant_client.__version__}")

except Exception as e:
    print(f"Error: {e}")
