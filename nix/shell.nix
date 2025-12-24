{pkgs, lib, config, ...}:
let
  py = pkgs.python313Packages;
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
      scikit-learn
      onnxruntime
      playwright
      rapidfuzz
      pydantic
      optimum
      pyexcel
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