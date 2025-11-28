import sys
import importlib

def check_dependencies():
    """Проверка наличия всех зависимостей"""
    dependencies = {
        "torch": "torch",
        "torchvision": "torchvision",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "pandas": "pandas",
        "PIL": "Pillow",
        "yaml": "PyYAML",
        "scipy": "scipy"
    }
    
    missing = []
    for import_name, package_name in dependencies.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(package_name)
            print(f"Отсутствует зависимость: {package_name}", file=sys.stderr)
    
    if missing:
        print("\nУстановите недостающие пакеты:", file=sys.stderr)
        print("   pip install " + " ".join(missing), file=sys.stderr)
        sys.exit(1)  # Прерываем выполнение

# Проверяем 
check_dependencies()
