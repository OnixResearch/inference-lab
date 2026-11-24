{
  description = "Inference Lab: reproducible inference, backend, and measurement experiments";

  inputs = {
    # Exact offline-compatible pins shared with OnixResearch/systems-lab.
    # Only the public compiler environment is shared; no product code is imported.
    nixpkgs.url = "github:NixOS/nixpkgs/ac588ef1c914865dad517b337ea0eb9694a4021c";
    rust-overlay = {
      url = "github:oxalica/rust-overlay/89750a2ef1c6a44db9f4251b16c23deffdcfddff";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, rust-overlay }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      pkgsFor = system: import nixpkgs {
        inherit system;
        overlays = [ rust-overlay.overlays.default ];
      };
      toolchainFor = pkgs: pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      platformFor = pkgs: pkgs.makeRustPlatform {
        cargo = toolchainFor pkgs;
        rustc = toolchainFor pkgs;
      };
    in {
      packages = forAllSystems (system:
        let
          pkgs = pkgsFor system;
          package = (platformFor pkgs).buildRustPackage {
            pname = "inference-lab";
            version = "0.1.0";
            src = self;
            cargoLock.lockFile = ./Cargo.lock;
            doCheck = true;
            cargoTestFlags = [ "--all-targets" ];
            meta = {
              description = "Bounded deterministic inference contract and measurement harnesses";
              mainProgram = "inference-lab";
            };
          };
          qwen38-benchmark = pkgs.callPackage ./pkgs/qwen38-benchmark { };
        in {
          default = package;
          inference-lab = package;
          inherit qwen38-benchmark;
        });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.default}/bin/inference-lab";
        };
        qwen38-benchmark = {
          type = "app";
          program = "${self.packages.${system}.qwen38-benchmark}/bin/qwen38-benchmark";
        };
      });

      devShells = forAllSystems (system:
        let pkgs = pkgsFor system;
        in {
          default = pkgs.mkShell {
            packages = [ (toolchainFor pkgs) pkgs.git pkgs.jq pkgs.ripgrep pkgs.nixfmt ];
          };
        });

      formatter = forAllSystems (system: (pkgsFor system).nixfmt);

      checks = forAllSystems (system:
        let pkgs = pkgsFor system;
        in {
          package-and-tests = self.packages.${system}.default;
          qwen38-benchmark = self.packages.${system}.qwen38-benchmark;
          rust-quality = pkgs.runCommand "inference-lab-rust-quality" {
            nativeBuildInputs = [ (toolchainFor pkgs) pkgs.stdenv.cc ];
          } ''
            cp -R ${self} source
            chmod -R u+w source
            cd source
            export CARGO_HOME="$TMPDIR/cargo-home"
            export CARGO_TARGET_DIR="$TMPDIR/target"
            mkdir -p "$CARGO_HOME"
            cargo fmt --all -- --check
            cargo clippy --offline --locked --all-targets -- -D warnings
            cargo test --offline --locked --doc
            touch "$out"
          '';
        });
    };
}
