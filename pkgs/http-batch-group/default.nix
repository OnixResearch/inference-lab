{
  lib,
  stdenvNoCC,
  python3,
  makeWrapper,
}:
stdenvNoCC.mkDerivation {
  pname = "http-batch-group";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ./.;
    filter = path: _type: baseNameOf path != "__pycache__";
  };

  nativeBuildInputs = [ makeWrapper ];
  buildInputs = [ python3 ];

  dontBuild = true;
  doCheck = true;
  doInstallCheck = true;

  checkPhase = ''
    runHook preCheck
    ${python3}/bin/python3 -m unittest discover -s tests -v
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin" "$out/libexec/http-batch-group" "$out/share/http-batch-group"
    install -m755 qwen38-batch-group.py "$out/libexec/http-batch-group/qwen38-batch-group.py"
    install -m644 workloads/qwen38-subagent-workload.json "$out/share/http-batch-group/qwen38-subagent-workload.json"
    install -m644 PROVENANCE.json "$out/share/http-batch-group/PROVENANCE.json"
    makeWrapper ${python3}/bin/python3 "$out/bin/http-batch-group" \
      --unset PYTHONPATH --unset PYTHONHOME \
      --add-flags "-I $out/libexec/http-batch-group/qwen38-batch-group.py"
    runHook postInstall
  '';

  installCheckPhase = ''
    runHook preInstallCheck
    "$out/bin/http-batch-group" --help >/dev/null
    test -f "$out/share/http-batch-group/qwen38-subagent-workload.json"
    runHook postInstallCheck
  '';

  meta = {
    description = "Barrier-synchronized OpenAI-compatible HTTP batch-group driver and frozen subagent workload";
    mainProgram = "http-batch-group";
    license = lib.licenses.lgpl21Only;
    platforms = lib.platforms.unix;
  };
}
