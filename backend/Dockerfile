# Dragonfly backend. Stages: deps -> dev (watch mode) -> build -> runtime (production deps only, non-root).
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM deps AS dev
COPY . .
EXPOSE 3000
CMD ["npm", "run", "start:dev"]

FROM deps AS build
COPY . .
RUN npm run build

FROM node:22-alpine AS runtime
ENV NODE_ENV=production
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --no-audit --no-fund && npm cache clean --force
COPY --from=build /app/dist ./dist
RUN addgroup -S dragonfly && adduser -S dragonfly -G dragonfly
USER dragonfly
EXPOSE 3000
CMD ["node", "dist/main.js"]
