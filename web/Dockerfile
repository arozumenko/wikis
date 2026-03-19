FROM node:20-alpine
WORKDIR /app
RUN apk add --no-cache curl openssl
COPY package*.json .
RUN npm ci
COPY prisma ./prisma
COPY scripts ./scripts
RUN chmod +x scripts/*.sh
COPY . .
RUN npx prisma generate
RUN npm run build
CMD ["./scripts/entrypoint.sh"]
EXPOSE 3000
