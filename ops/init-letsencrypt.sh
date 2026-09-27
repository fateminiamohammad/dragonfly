#!/usr/bin/env bash
# Issue the first Let's Encrypt certificate for APP_DOMAIN. After this, the certbot container renews it.
#   APP_DOMAIN=decide.example.com LETSENCRYPT_EMAIL=you@example.com ops/init-letsencrypt.sh
set -euo pipefail
: "${APP_DOMAIN:?set APP_DOMAIN}"
: "${LETSENCRYPT_EMAIL:?set LETSENCRYPT_EMAIL}"
COMPOSE=(docker compose -f docker-compose.prod.yml --env-file "docker/${ENV_FILE_NAME:-.env}" --project-name dragonfly)

# nginx needs a certificate file to start: create a throwaway self-signed one first
"${COMPOSE[@]}" run --rm --entrypoint sh certbot -c "
  mkdir -p /etc/letsencrypt/live/$APP_DOMAIN &&
  openssl req -x509 -nodes -newkey rsa:2048 -days 1 -subj /CN=localhost \
    -keyout /etc/letsencrypt/live/$APP_DOMAIN/privkey.pem -out /etc/letsencrypt/live/$APP_DOMAIN/fullchain.pem"
"${COMPOSE[@]}" up -d nginx

# replace it with a real certificate via the webroot challenge
"${COMPOSE[@]}" run --rm --entrypoint sh certbot -c "rm -rf /etc/letsencrypt/live/$APP_DOMAIN /etc/letsencrypt/archive/$APP_DOMAIN /etc/letsencrypt/renewal/$APP_DOMAIN.conf"
"${COMPOSE[@]}" run --rm --entrypoint certbot certbot certonly --webroot -w /var/www/certbot \
  -d "$APP_DOMAIN" --email "$LETSENCRYPT_EMAIL" --agree-tos --no-eff-email
"${COMPOSE[@]}" exec nginx nginx -s reload
echo "certificate issued for $APP_DOMAIN"
