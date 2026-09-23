# Multi-platform image digests verified on 2026-09-23.
FROM node:24-alpine@sha256:ebfe2f90462722a7a4de65e91990e97fe0d401c70e0e762c5b53302f905ec1c1 AS builder
WORKDIR /application/user_interface
COPY user_interface/package.json user_interface/package-lock.json ./
RUN npm ci
COPY user_interface/ ./
COPY contracts/ /application/contracts/
RUN npm run build

FROM nginxinc/nginx-unprivileged:stable-alpine@sha256:4714e0b1b2577eaa1a6131d07c958b67f0eb68e6d0521e90c6e5287db8cf0bc5
COPY user_interface/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /application/user_interface/dist/ /usr/share/nginx/html/
USER 101:101
EXPOSE 8080
ENTRYPOINT ["nginx"]
CMD ["-g", "daemon off;"]
