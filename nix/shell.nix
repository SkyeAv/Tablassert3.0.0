{pkgs, lib, config, ...}:
let
  py = pkgs.python313Packages.override {
    overrides = self: super: {
      optimum-onnx = self.buildPythonPackage rec {
        pname = "optimum-onnx";
        version = "0.1.0";
        format = "pyproject";
        src = pkgs.fetchFromGitHub {
          owner = "huggingface";
          repo = "optimum-onnx";
          rev = "v0.1.0-release";
          sha256 = "sha256-Thx3QPLgi8w8znvMGSuCyRu/tUynCkQFywtKKv7UhuA=";
        };
        build-system = (with self; [
          setuptools
          wheel
        ]);
        propagatedBuildInputs = (with self; [
          transformers
          scikit-learn
          onnxruntime
          optimum
          scipy
          onnx
        ]);
        passthru.optional-dependencies.onnxruntime = [self.onnxruntime];
        pythonRelaxDeps = ["optimum"];
        doCheck = false;
      };
    };
  };
  
  tablassert = py.buildPythonApplication rec {
    pname = "tablassert";
    version = "6.0.0";
    format = "pyproject";
    src = ../.;
    build-system = (with py; [
      setuptools
      wheel
    ]);
    propagatedBuildInputs = (with py; [
      sentence-transformers
      optimum-onnx
      scikit-learn
      sqlite-utils
      onnxruntime
      playwright
      rapidfuzz
      diskcache
      pydantic
      pyexcel
      mkdocs
      pyyaml
      duckdb
      orjson
      polars
      typer
    ]) ++ (with pkgs; [
      chromium
      gawk
      jq
    ]);
    nativeBuildInputs = (with pkgs; [
      makeWrapper
    ]);
    makeWrapperArgs = [
      "--set CHROMIUM_PATH ${pkgs.chromium}/bin/chromium"
      "--set PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD 1"
      "--set AWK_PATH ${pkgs.gawk}/bin/gawk"
      "--set JQ_PATH ${pkgs.jq}/bin/jq"
    ];
    doCheck = false;
  };
in {
  devShells.default = pkgs.mkShell {
    packages = (with py; [
      python
      flake8
    ]) ++ ([
      tablassert
    ]);
  };
}