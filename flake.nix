{
  description = "lbfgs";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:nixos/nixpkgs/master";
    foolnotion.url = "github:foolnotion/nur-pkg";
    foolnotion.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, flake-utils, nixpkgs, foolnotion }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ foolnotion.overlay ];
        };
        stdenv_ = pkgs.llvmPackages_16.stdenv;
      in rec {
          devShells.default = stdenv_.mkDerivation {
            name = "lbfgs dev";

            nativeBuildInputs = with pkgs; [
              cmake
              clang-tools_16
              cppcheck
              include-what-you-use
              cmake-language-server
            ];

            buildInputs = with pkgs; [
              bear
              eigen
              gdb
              graphviz
              hyperfine
              linuxPackages_latest.perf
              valgrind
              ned14-outcome
              ned14-quickcpplib
              ned14-status-code
              libnano
            ];

            shellHook = ''
              alias bb="cmake --build build -j"
            '';
          };
      });
}
