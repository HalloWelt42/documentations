
```git
# -------------------------------------------------
# macOS / Apple‑Silicon (M‑Chip) – systembezogene Dateien
# -------------------------------------------------
.DS_Store                 # Finder‑Cache‑Dateien
.AppleDouble              # Metadaten für Netzwerk‑Shares
.LSOverride               # LaunchServices‑Overrides

# Icon‑ und Thumbs‑Dateien
Icon?                     # benutzerdefinierte Ordnersymbole
._*                       # resource‑fork‑Begleiterdateien
Thumbs.db                 # (kommt von Windows, aber häufig in gemounteten Volumes)

# Spotlight / Finder‑Spezialordner
.Spotlight-V100
.Trashes
.VolumeIcon.icns

# -------------------------------------------------
# Python – Bytecode, Build‑Artefakte, virtuelle Umgebungen …
# -------------------------------------------------
# Byte‑compiled / optimierte Dateien
__pycache__/              # Zwischenspeicher für kompilierten Code
*.py[cod]                 # *.pyc, *.pyo, *.pyd
*$py.class

# C‑Extensions (z. B. .so‑Dateien)
*.so

# Distribution / Packaging
build/
dist/
*.egg-info/
.eggs/
pip-wheel-metadata/
*.egg# Installer‑Logs
pip-log.txt
pip-delete-this-directory.txt

# Unit‑Test‑ und Coverage‑Reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
pytestdebug.log

# Jupyter Notebook – Checkpoints
.ipynb_checkpoints/

# Type‑Checker (mypy / pyre)
.mypy_cache/
.dmypy.json
dmypy.json
.pyre/

# Virtual Environments & Conda
.env
.venv/
env/
venv/
ENV/
virtualenv/
__pypackages__/

# ---------------------------------------------------------
# IDEs / Editoren – PyCharm, VS Code, … 
# ---------------------------------------------------------
# PyCharm 
#  (empfohlen: 
#    .idea komplett ignorieren; wenn du bestimmte Settings teilen willst,
#     dann einzelne Dateien auskommentieren bzw. whitelist‑en)
.idea/
*.iml
*.iws
*.ipr

# Visual Studio Code (falls du es gelegentlich nutzt)
.vscode/

# Sublime Text
*.sublime-workspace
*.sublime-project

# Emacs
*~
\#*\#
.#*

# Vim
*.swp
*.swo
.session

# ---------------------------------------------------------
# Sonstige temporäre / generierte Dateien
# ---------------------------------------------------------
logs/
*.log                     # Log‑Dateien

# Datenbank‑Dateien (z. B. SQLite)
*.sqlite3
*.db

# Generierte Mediendateien (falls du z. B. Bilder/Assets im Build‑Prozess erzeugst)
media/

# ---------------------------------------------------------
# Optional: Dateien, die du bewusst versionieren willst,
# auskommentieren oder entfernen.
# ---------------------------------------------------------
# .idea/runConfigurations.xml   # Beispiel: Run‑Konfigurationen teilen
# .env.example                  # Vorlage für Umgebungsvariablen 
#                                 (nicht das eigentliche .env)

```
