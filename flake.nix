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
        packages.default = stdenv_.mkDerivation {
          name = "lbfgs";
          src = self;

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DCMAKE_CXX_FLAGS=-march=x86-64-v3"
          ];
          nativeBuildInputs = with pkgs; [
            cmake
          ];

          buildInputs = with pkgs; [
            eigen
            ned14-outcome
            ned14-quickcpplib
            ned14-status-code
          ];
        };

        devShells.default = stdenv_.mkDerivation {
          name = "lbfgs dev";

          nativeBuildInputs = packages.default.nativeBuildInputs ++ (with pkgs; [
            clang-tools_16
            cppcheck
            include-what-you-use
            cmake-language-server
          ]);

          buildInputs = packages.default.buildInputs ++ (with pkgs; [
            gdb
            linuxPackages_latest.perf
            valgrind
            libnano
          ]);

          shellHook = ''
            alias bb="cmake --build build -j"
          '';
        };
      });
}
