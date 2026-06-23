#!/bin/bash
set -e

mkdir -p /jobs/logs /jobs/models /jobs/tb_28may

cd /usr/src
exec "$@"
