# Zero-Downtime Migration Guide

## Migrating to Sharded Collections

Qdrant does not support online resharding. Changing `shard_number` requires recreating the collection. Use this guide to migrate without downtime.

### Migration Steps

1. **Create new sharded collection** (e.g., `documents_v2`):
   ```bash
   curl -X PUT http://localhost:6333/collections/documents_v2 \
     -H "Content-Type: application/json" \
     -d '{
       "vectors": {"size": 384, "distance": "Dot", "on_disk": true},
       "shard_number": 6,
       "replication_factor": 1
     }'
   ```

2. **Dual-write during cutover period**:
   - Modify ingestion to write to both `documents` and `documents_v2`
   - Run for a period equal to your max ingestion lag

3. **Backfill existing data**:
   ```python
   # Use scroll API to read from old collection and upsert to new
   from qdrant_client import QdrantClient

   client = QdrantClient("localhost", port=6333)
   offset = None
   while True:
       result = client.scroll(
           collection_name="documents",
           offset=offset,
           limit=1000,
           with_vectors=True,
       )
       points = result[0]
       if not points:
           break
       client.upsert(collection_name="documents_v2", points=points)
       offset = result[1]
   ```

4. **Update application config**:
   ```env
   QDRANT_COLLECTION=documents_v2
   ```

5. **Delete old collection** after verifying new collection:
   ```bash
   curl -X DELETE http://localhost:6333/collections/documents
   ```

### Rollback Plan

If issues are detected after cutover:
1. Revert `QDRANT_COLLECTION` to `documents`
2. The old collection remains intact until explicitly deleted
