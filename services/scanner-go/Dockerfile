
FROM golang:1.22-alpine

WORKDIR /app

COPY . .

WORKDIR /app/services/scanner-go
RUN go mod download
RUN go build -o /scanner .

CMD ["/scanner"]
