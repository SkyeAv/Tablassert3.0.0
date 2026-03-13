{pkgs, lib, config, ...}: 
let 
  py = pkgs.python313Packages;
in {
  packages.default = py.tablassert;
  devShells.default = pkgs.mkShell {
    packages = (with py; [
      tablassert
      pytest
      flake8
      mkdocs
    ]) ++ (with pkgs; [
      pyright
      pylint
      ruff
    ]);
  };
}