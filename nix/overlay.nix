final: prev: {
  python313Packages = prev.python313Packages.override {
    overrides = pyFinal: pyPrev: {
      optimum-onnx = pyFinal.buildPythonPackage rec {
        pname = "optimum-onnx";
        version = "0.1.0";
        format = "pyproject";
        src = final.fetchFromGitHub {
          owner = "huggingface";
          repo = "optimum-onnx";
          rev = "v0.1.0-release";
          sha256 = "sha256-Thx3QPLgi8w8znvMGSuCyRu/tUynCkQFywtKKv7UhuA=";
        };
        build-system = with pyFinal; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = with pyFinal; [
          transformers
          scikit-learn
          onnxruntime
          optimum
          scipy
          onnx
        ];
        passthru.optional-dependencies.onnxruntime = [pyFinal.onnxruntime];
        pythonRelaxDeps = ["optimum"];
        doCheck = false;
      };
      tablassert = pyFinal.buildPythonApplication rec {
        pname = "tablassert";
        version = "6.1.0";
        format = "pyproject";
        src = ../.;
        build-system = with pyFinal; [
          setuptools
          wheel
        ];
        propagatedBuildInputs = (with pyFinal; [
          sentence-transformers
          optimum-onnx
          scikit-learn
          sqlite-utils
          onnxruntime
          playwright
          rapidfuzz
          diskcache
          fastexcel
          pydantic
          pyexcel
          pyarrow
          mkdocs
          pyyaml
          duckdb
          orjson
          polars
          typer
        ]) ++ (with final; [
          chromium
          gawk
          jq
        ]);
        nativeBuildInputs = [final.makeWrapper];
        makeWrapperArgs = [
          "--set CHROMIUM_PATH ${final.chromium}/bin/chromium"
          "--set PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD 1"
          "--set AWK_PATH ${final.gawk}/bin/gawk"
          "--set JQ_PATH ${final.jq}/bin/jq"
        ];
        doCheck = false;
      };
    };
  };
}