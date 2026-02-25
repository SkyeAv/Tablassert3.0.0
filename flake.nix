{
  description = "tablassert (6.2.0)";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };
  outputs = inputs @ {self, nixpkgs, flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {pkgs, lib, config, system, ...}: {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          overlays = [self.overlays.default];
        };
        imports = [
          ./nix/docker.nix
          ./nix/shell.nix
        ];
      };
      flake = {
        overlays.default = import ./nix/overlay.nix;
      };
    };
}