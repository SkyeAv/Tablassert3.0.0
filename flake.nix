{
  description = "tablassert (6.1.0)";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };
  outputs = inputs @ {self, systems, nixpkgs, flake-parts, ...}:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = import inputs.systems;
      perSystem = {pkgs, lib, config, system, ...}: {
        _module.args.pkgs = import nixpkgs {
          inherit system;
          overlays = [self.overlays.default];
        };
        imports = [
          ./nix/shell.nix
        ];
      };
      flake = {
        overlays.default = import ./nix/overlay.nix;
      };
    };
}