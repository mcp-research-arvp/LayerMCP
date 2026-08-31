#!/bin/bash
exec bash "$(dirname "$0")/submit_all.sh" --multi-only "$@"
