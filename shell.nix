{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python313;

  # Custom Python packages not in nixpkgs
  tmmFast = pkgs.python313Packages.buildPythonPackage rec {
    pname = "tmm-fast";
    version = "0.2.1";

    src = pkgs.fetchPypi {
      pname = "tmm_fast";
      version = "0.2.1";
      sha256 = "de9a4521d8bd2cbc37026d25ecb084fa9855236013e4faea28ee8be9c39ef650";
    };

    # 🔑 THIS is what you're missing
    pyproject = true;

    # Build system (PEP 517)
    nativeBuildInputs = with pkgs.python313Packages; [
      setuptools
      wheel
      setuptools-scm
    ];

    propagatedBuildInputs = with pkgs.python313Packages; [
      numpy
      torch
      matplotlib
      gymnasium
    ];

    meta = with pkgs.lib; {
      description = "TMM-fast library";
      license = licenses.mit;
    };
  };

in pkgs.mkShell {
  buildInputs = [
    python
    pkgs.python313Packages.numpy
    pkgs.python313Packages.scipy
    pkgs.python313Packages.pandas
    pkgs.python313Packages.matplotlib
    pkgs.python313Packages.torch
    pkgs.python313Packages.seaborn
    pkgs.python313Packages.ipykernel  # for VS Code / Jupyter
    pkgs.python313Packages.jupyterlab # optional
    tmmFast
    pkgs.gcc
    pkgs.zlib
    pkgs.bzip2
    pkgs.openssl
    pkgs.libpng
    pkgs.libjpeg
    pkgs.ffmpeg
    pkgs.libxkbcommon
    pkgs.fontconfig
    pkgs.freetype
    pkgs.zstd
    pkgs.dbus
    pkgs.xorg.libXrender
    pkgs.xorg.libxcb
    pkgs.xorg.libX11
    pkgs.xorg.libXext
    pkgs.xorg.libXcursor
  ];

  # optional: helpful environment info
  shellHook = ''
    echo "Python environment active"
    python --version
  '';
}