#!/bin/sh
set -e

: "${PORT:=80}"
if [ -f /etc/nginx/nginx.conf.template ]; then
  envsubst '${PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf
fi
nginx -g 'daemon off;'
