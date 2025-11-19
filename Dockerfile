# Stage 1: Build Hugo site
FROM hugomods/hugo:exts AS builder

# Set working directory
WORKDIR /src

# Copy the entire site
COPY . .

# Build the Hugo site
RUN hugo --minify --baseURL https://perlou.top/

# Stage 2: Serve with Nginx
FROM nginx:alpine

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Copy the built site from builder stage
COPY --from=builder /src/public /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
