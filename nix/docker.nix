{pkgs, lib, config, ...}: 
let 
  py = pkgs.python313Packages;
in {
  packages.docker = pkgs.dockerTools.buildImage {
    name = "tablassert-cli";
    tag = "latest";
    copyToRoot = pkgs.buildEnv {
      name = "image-root";
      paths = (with py; [
        tablassert
      ]);
      pathsToLink = [
        "/bin"
        "/etc"
      ];
    };
    config = {
      Entrypoint = ["tablassert-cli"];
      WorkingDir = "/workdir";
    };
  };
}