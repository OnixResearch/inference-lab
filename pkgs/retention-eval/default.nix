{
  lib,
  stdenvNoCC,
  makeWrapper,
  python3,
  withMlDeps ? false,
}:
let
  python =
    if withMlDeps then
      python3.withPackages (p: [
        p.torch
        p.transformers
      ])
    else
      python3;
in
stdenvNoCC.mkDerivation {
  pname = if withMlDeps then "retention-eval-full" else "retention-eval";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ./.;
    filter = path: _type: baseNameOf path != "__pycache__";
  };

  nativeBuildInputs = [ makeWrapper ];

  dontBuild = true;
  doCheck = true;
  doInstallCheck = true;

  checkPhase = ''
    runHook preCheck
    ${lib.optionalString withMlDeps "REQUIRE_TORCH=1 "}${python}/bin/python -m unittest discover -s tests -v
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/share/retention-eval" "$out/bin"
    cp -r cli.py retention.py data PROVENANCE.json "$out/share/retention-eval/"
    makeWrapper ${python}/bin/python "$out/bin/retention-eval" \
      --unset PYTHONPATH --unset PYTHONHOME \
      --add-flags "-I $out/share/retention-eval/cli.py"
    runHook postInstall
  '';

  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/retention-eval" --help >/dev/null
    "$out/bin/retention-eval" compare --help >/dev/null
    runHook postInstallCheck
  '';

  meta = {
    description = "Provenance-bound six-passage full-vocabulary retention evaluator";
    license = lib.licenses.lgpl21Only;
    mainProgram = "retention-eval";
    platforms = lib.platforms.unix;
  };
}
