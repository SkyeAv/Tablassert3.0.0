{
  description = "TABLASSERT";
  inputs = {
    nixpkgs = {
      url = "github:nixos/nixpkgs/25.05";
    };
  };
  outputs = {
    self,
    nixpkgs
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
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-623xLK2bf7Vr8BpzemcMSk5n0yD2VjBZ4vjsy6OQTX0=";
        };
        build-system = with py; [
          setuptools
          setuptools-scm
          wheel
        ];
        propagatedBuildInputs = with py; [
          torch
          numpy
          scikit-learn
        ];
        doCheck = false;
      };
      hBreader = py.buildPythonPackage rec {
        pname = "hbreader";
        version = "0.9.1";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-0sEy+LpidteUxmIkwyl87CXIB50KTPAZwGFhHgo7lPo=";
        };
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = with py; [
          pbr
        ];
        doCheck = false;
      };
      jsonFlattener = py.buildPythonPackage rec {
        pname = "json_flattener";
        version = "0.1.9";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-hM+FIwRf+xJDAaYCYCIBZl/LADoXHs6H5vRu0C9/DBU=";
        };
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = with py; [
          click
          pyyaml
        ];
        doCheck = false;
      };
      jsonAsObj2 = py.buildPythonPackage rec {
        pname = "jsonasobj2";
        version = "1.0.4";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-9QsWaO9HgASqSHstLQlMME5ctreTN4CfSh8pdcx/u04=";
        };
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = [
          py.pbr
          hBreader
        ];
        doCheck = false;
      };
      pytestLogging = py.buildPythonPackage rec {
        pname = "pytest-logging";
        version = "2015.11.4";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-zsXIXs8Yqrey6tVJijG591hoDvWpArkFSrPyvbt3yJY=";
        };
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = with py; [
          pytest
        ];
        doCheck = false;
      };
      prefixCommons = py.buildPythonPackage rec {
        pname = "prefixcommons";
        version = "0.1.12";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-IsTi03tjSHs6tI8ElbcPFFZMs0ahUiDyORnrDBhR9p8=";
        };
        build-system = with py; [
          poetry-core
          poetry-dynamic-versioning
          wheel
        ];
        propagatedBuildInputs = [
          py.pyyaml
          py.click
          py.requests
          pytestLogging
        ];
        doCheck = false;
      };
      pyTrie = py.buildPythonPackage rec {
        pname = "PyTrie";
        version = "0.4.0";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-j0SI9ALTRlmT+2tu+gmGaEntjNp5A7UGR7fQNCuAU3k=";
        };
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = with py; [
          sortedcontainers
        ];
        doCheck = false;
      };
      curies = py.buildPythonPackage rec {
        pname = "curies";
        version = "0.10.19";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-rq5efLt67mxRRDdvy2nhWg08BVehL57f+Am9DOUATqI=";
        };
        build-system = with py; [
          hatchling
          wheel
        ];
        propagatedBuildInputs = [
          py.pydantic
          pyTrie
          py.typing-extensions
        ];
        doCheck = false;
      };
      prefixMaps = py.buildPythonPackage rec {
        pname = "prefixmaps";
        version = "0.2.6";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-dCHhJE7qYQIX+hupbJrr1k6BYqkw3AYmIHzYv2Ls9Lk=";
        };
        build-system = with py; [
          poetry-core
          poetry-dynamic-versioning
          wheel
        ];
        propagatedBuildInputs = [
          curies
          py.pyyaml
        ];
        doCheck = false;
      };
      linkmlRuntime = py.buildPythonPackage rec {
        pname = "linkml_runtime";
        version = "1.9.4";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-rI8Bqk+S6zLKN3vxXOQtrWP8LSIB4I4rlViW40AWB1s=";
        };
        build-system = with py; [
          poetry-core
          poetry-dynamic-versioning
          wheel
        ];
        propagatedBuildInputs = [
          py.deprecated
          py.jsonschema
          py.pydantic
          py.pyyaml
          py.rdflib
          py.requests
          hBreader
          jsonFlattener
          jsonAsObj2
          prefixCommons
          prefixMaps
        ];
        doCheck = false;
      };
      bmt = py.buildPythonPackage rec {
        pname = "bmt";
        version = "1.4.5";
        pyproject = true;
        src = pkgs.fetchPypi {
          inherit pname version;
          sha256 = "sha256-hHOG6DssXU5iBQrUozbcXAQzMFk7jQchbFsY/rp2PiM=";
        };
        build-system = with py; [
          poetry-core
          poetry-dynamic-versioning
          wheel
        ];
        propagatedBuildInputs = [
          py.deprecation
          linkmlRuntime
          py.stringcase
        ];
        doCheck = false;
      };
    in {
      myapp = py.buildPythonApplication {
        pname = "tablassert";
        version = "4.4.0";
        src = ./.;
        pyproject = true;
        build-system = with py; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = [
          pkgs.chromium
          py.pydantic
          py.ruamel-yaml
          py.sqlite-utils
          py.diskcache
          py.loguru
          py.spacy
          py.polars
          py.xlsx2csv
          py.typer
          py.deepmerge
          py.pyarrow
          py.requests
          py.openpyxl
          py.xlrd
          py.pandas
          py.torch
          py.transformers
          py.scikit-learn
          py.numpy
          py.joblib
          torchDr
          bmt
          py.matplotlib
          py.playwright
          enCoreWebSm
        ];
        nativeBuildInputs = [
          pkgs.makeWrapper
        ];
        makeWrapperArgs = [
          "--set CHROMIUM_PATH ${pkgs.chromium}/bin/chromium"
          "--set PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD 1"
        ];
      };
      default = self.packages.${system}.myapp;
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
