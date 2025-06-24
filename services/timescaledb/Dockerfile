# Use the official TimescaleDB image as a base
FROM timescale/timescaledb-ha:pg16-ts2.14

# Switch to root user to install packages
USER root

# Add PostgreSQL APT repository and install pgvector
# See https://wiki.postgresql.org/wiki/Apt
# And https://github.com/pgvector/pgvector#apt
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        lsb-release \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc | gpg --dearmor -o /usr/share/keyrings/postgresql-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql-archive-keyring.gpg] http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the default postgres user
USER postgres
