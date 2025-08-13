{
  description = "TABLASSERT";
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/25.05";
    };
    tablassert = {
      url = "github:SkyeAv/Tablassert/4.4.0";
      flake = false;
    };
  };
  outputs = {
    self,
    nixpkgs,
    tablassert,
  }: let
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
          };
          lib = nixpkgs.lib;
        });
  in {
    packages = forAllSystems ({
      pkgs,
      lib,
      system,
    }: let
      py = pkgs.python312Packages;
      enCoreWebSm = py.buildPythonPackage rec {
        pname = "en-core-web-sm";
        version = "3.8.0";
        format = "wheel";
        src = pkgs.fetchurl {
          url = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-${version}/en_core_web_sm-${version}-py3-none-any.whl";
          sha256 = "sha256-GTJCnbcn1L/z3u1rNM/AXfF3lPSlLusmz4ko98Gg+4U=";
        };
        propagatedBuildInputs = [
          py.spacy
        ];
        doCheck = false;
      };
      torchDr = py.buildPythonPackage rec {
        pname = "torchdr";
        version = "0.3";
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-623xLK2bf7Vr8BpzemcMSk5n0yD2VjBZ4vjsy6OQTX0=";
        };
        doCheck = false;
      };
      bmt = py.buildPythonPackage rec {
        pname = "bmt";
        version = "1.4.5";
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-hHOG6DssXU5iBQrUozbcXAQzMFk7jQchbFsY/rp2PiM=";
        };
        doCheck = false;
      };
    in {
      myapp = py.buildPythonApplication {
        pname = "tablassert";
        version = "4.4.0";
        src = tablassert;
        pyproject = true;
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = with py; [
          pydantic
          ruamel-yaml
          sqlite-utils
          diskcache
          loguru
          spacy
          polars
          xlsx2csv
          typer
          deepmerge
          pyarrow
          requests
          openpyxl
          xlrd
          pandas
          torch
          transformers
          scikit-learn
          numpy
          joblib
          torchDr
          bmt
          matplotlib
          playwright
          enCoreWebSm
        ];
      };
      default = self.packages.${system}.myapp;
    });
    devShells = forAllSystems ({
      pkgs,
      system,
      ...
    }: {
      default = pkgs.mkShell {
        packages = [
          self.packages.${system}.myapp-env
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
    apps = forAllSystems ({
      pkgs,
      system,
      ...
    }: {
      default = {
        type = "app";
        program = "${self.packages.${system}.myapp}/bin/cli";
      };
    });
  };
}
