# BUILD
FROM nixos/nix:2.23.1 AS builder
WORKDIR /src

ENV NIX_CONFIG="experimental-features = nix-command flakes"

COPY flake.nix flake.lock* ./

RUN nix flake show

ARG SYSTEM=x86_64-linux
RUN nix build .#packages.${SYSTEM}.myapp

RUN mkdir -p /closure && \
    nix-store -qR --include-outputs ./result | xargs -I{} cp -r --no-preserve=ownership --parents {} /closure && \
    cp -r --no-preserve=ownership result /closure/app

# RUNTIME
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /closure/ /

RUN useradd -u 10001 -m app
USER app

ENV PATH="/app/bin:${PATH}"

EXPOSE 8080

ENTRYPOINT ["cli"]
CMD ["--help"]