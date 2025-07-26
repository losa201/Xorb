FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install git for go modules
RUN apk add --no-cache git ca-certificates

# Copy go modules first for better caching
COPY go.mod ./
RUN go mod download

# Copy source code
COPY . .

# Tidy modules and build the application
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o scanner .

# Runtime stage
FROM alpine:3.18

# Install nuclei and required tools
RUN apk add --no-cache curl ca-certificates wget \
    && curl -sSfL https://github.com/projectdiscovery/nuclei/releases/download/v3.0.4/nuclei_3.0.4_linux_amd64.zip -o nuclei.zip \
    && unzip nuclei.zip \
    && mv nuclei /usr/local/bin/ \
    && rm nuclei.zip \
    && nuclei -update-templates

# Create app directory and user
RUN adduser -D -s /bin/sh scanner
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/scanner .

# Set ownership
RUN chown -R scanner:scanner /app
USER scanner

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8004/health || exit 1

# Expose port
EXPOSE 8004

# Run the scanner service
CMD ["./scanner"]