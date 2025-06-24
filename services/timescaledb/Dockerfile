# Use the official pgvector image for PostgreSQL 17 as the source for the extension files
FROM pgvector/pgvector:pg17 as pgvector

# Use the TimescaleDB image for PostgreSQL 17 as the base
FROM timescale/timescaledb-ha:pg17

# Copy the pgvector extension files from the pgvector image
COPY --from=pgvector /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/
COPY --from=pgvector /usr/share/postgresql/17/extension/vector--*.sql /usr/share/postgresql/17/extension/
COPY --from=pgvector /usr/share/postgresql/17/extension/vector.control /usr/share/postgresql/17/extension/

# Switch back to the default postgres user
USER postgres
