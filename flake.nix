{
  description = "Python development environment with virtualenv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python3;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pkgs.nushell
            pkgs.direnv
            pkgs.stdenv.cc.cc.lib
            pkgs.glibc
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.stdenv.cc.cc.lib
            pkgs.glibc
            pkgs.libz
          ];

          shellHook = ''
            # Set up virtualenv if it doesn't exist
            if [ ! -d .venv ]; then
              echo "Creating virtual environment..."
              ${python}/bin/python -m venv .venv
            fi

            # Activate virtualenv
            source .venv/bin/activate

            # Install requirements if requirements.txt exists
            if [ -f requirements.txt ]; then
              echo "Installing requirements..."
              pip install -r requirements.txt
            fi

            echo "Python environment ready!"
            echo "Python version: $(python --version)"
            echo "Virtual environment: $VIRTUAL_ENV"
          '';
        };
      }
    );
}
