{
  packageName = "tablassert";
  authors = [
    "Skye Lane Goetz"
    "Gwenlyn Glusman"
  ];
  description = "Tablassert is a versatile tool that creates knowledge assertions from tabular data, enhances knowledge with configurable options, and exports KGX-compliant TSVs";
  homepage = "https://github.com/SkyeAv/Tablassert";
  version = "4.4.0";
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/25.05";
    };
    tablassert = {
      url = "github:SkyeAv/Tablassert/4.4.0";
      flake = false;
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
    };
  };
  outputs = {self, nixpkgs, tablassert, poetry2nix}:
  let 
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
      ];
    forAllSystems = f:
      nixpkgs.lib.genAttrs systems (system:
        f {
          inherit system;
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              poetry2nix.overlay
              ];
          };
          lib  = nixpkgs.lib;
        });
  in {
    packages = forAllSystems ({pkgs, lib, system}:
      let
        python = pkgs.python313;
        py = pkgs.python313Packages;
      in {
        myapp = pkgs.poetry2nix.mkPoetryApplication {
          projectDir = tablassert;
          python = pkgs.python313;
        };
        python-env = pkgs.poetry2nix.mkPoetryEnv {
          projectDir = tablassert;
          python = pkgs.python313;
        };
        default = self.packages.${system}.myapp;
      });
    devShells = forAllSystems ({pkgs, system, ...}: {
      default = pkgs.mkShell {
        packages = [
          self.packages.${system}.python-env
          pkgs.git
          pkgs.pkg-config
          pkgs.playwright-core
        ];
        shellHook = ''
          echo "Dev shell for ${system}"
          python3 --version
        '';
      };
    });
    apps = forAllSystems ({pkgs, system, ...}: {
      default = {
        type = "app";
        program = "${self.packages.${system}.myapp}/bin/cli";
      };
    });
  }
}
