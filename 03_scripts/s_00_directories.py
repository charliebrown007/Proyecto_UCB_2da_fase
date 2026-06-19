from pathlib import Path

# 1. Definir la estructura (la que ya usabas en R, que es excelente)

directories = [
    '00_raw_data',
    '01_cleaned_data',
    '02_environments', 
    '03_scripts',
    '04_tables',
    '05_graphs',
    '06_quarto'
]

# 2. Crear las carpetas

for folder in directories:
    Path(folder).mkdir( parents = True, exist_ok = True)

# 3. .gitignore

gitignore = Path(".gitignore")
if not gitignore.exists():
    content = [
        '__pycache__/',        # Basura de ejecución de Python
        '.ipynb_checkpoints/', # Temporales de Jupyter
        '00_raw_data/*',       # Omitir datos crudos en git
        '02_environments/*',   # El entorno virtual suele ser pesado
        '*.pyc'
    ]
    gitignore.write_text("\n".join(content))
    print('✅ .gitignore creado')

# 4. Un README.md minimalista
readme = Path('README.md')
if not readme.exists():
    project_name = Path.cwd().name.replace('_', ' ').upper()
    content = f"# {project_name}\n\nDescripción breve del proyecto..."
    readme.write_text(content, encoding = 'utf-8')
    print('✅ README.md básico creado')

print(f"\n📂 ¡Estructura lista para {Path.cwd().name.replace('_', ' ').upper()}")