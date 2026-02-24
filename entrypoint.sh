#!/bin/sh
set -e

: "${PORT:=80}"

# If a template exists, substitute PORT into it and write nginx default
if [ -f /etc/nginx/sites-available/default.template ]; then
  envsubst '${PORT}' < /etc/nginx/sites-available/default.template > /etc/nginx/sites-available/default
fi

# Start supervisord (will bring up nginx + services)
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
