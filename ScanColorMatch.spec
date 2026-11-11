# PyInstaller recipe: one folder holding the window program and a console program.
#   ScanColorMatch.exe      the interface (no console window)
#   ScanColorMatch-cli.exe  the command line (output in the terminal, real exit codes)
# Both share one set of libraries, so the second program costs a few MB.
a = Analysis(
    ["scan_color_match.py", "scan_color_match_cli.py"],
    pathex=["."],
    # The program's own modules sit next to the entry scripts; name them explicitly.
    hiddenimports=["engine", "i18n", "project", "calibration_worker"],
    # The colour engine loads this file by path, so it ships as a file too.
    datas=[("match_scan_to_reference.py", ".")],
)
pyz = PYZ(a.pure)

ENTRY = ("scan_color_match", "scan_color_match_cli")


def scripts_for(entry):
    """The entry script itself, plus the runtime hooks every program needs."""
    return [item for item in a.scripts if item[0] == entry or item[0] not in ENTRY]


gui = EXE(pyz, scripts_for("scan_color_match"), [], exclude_binaries=True,
          name="ScanColorMatch", console=False)
cli = EXE(pyz, scripts_for("scan_color_match_cli"), [], exclude_binaries=True,
          name="ScanColorMatch-cli", console=True)
coll = COLLECT(gui, cli, a.binaries, a.datas, name="ScanColorMatch")
