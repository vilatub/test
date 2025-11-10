# Настройка для Windows

Этот документ описывает настройку репозитория для работы на Windows, особенно при работе с Jupyter notebooks.

## Проблема: Line Endings (CRLF vs LF)

**Проблема:** Jupyter на Windows сохраняет файлы `.ipynb` с Windows line endings (`\r\n` = CRLF), что создает проблемы:
- Диффы в git выглядят сломанными (показывают изменения во всех строках)
- Текстовые редакторы показывают `^M` символы
- Проблемы при работе в команде (Windows + Linux/Mac)

**Решение:** Принудительное использование Unix line endings (LF) для всех текстовых файлов.

## Что уже настроено

### 1. `.gitattributes`

Файл `.gitattributes` автоматически конвертирует line endings при checkout/commit:

```
*.ipynb text eol=lf
*.py text eol=lf
*.md text eol=lf
```

Это заставляет git всегда хранить эти файлы с LF, независимо от вашей ОС.

### 2. Pre-commit Hook

Git hook `.git/hooks/pre-commit` автоматически исправляет CRLF → LF перед каждым коммитом.

**Установка hook (если еще не установлен):**

```bash
# Скопировать из репозитория
cp hooks/pre-commit .git/hooks/pre-commit

# Сделать исполняемым (Linux/Mac)
chmod +x .git/hooks/pre-commit
```

**Что делает hook:**

- Проверяет все `.ipynb` файлы
- Если находит CRLF, автоматически конвертирует в LF
- Повторно добавляет исправленные файлы в staging

## Первоначальная настройка на Windows

### Шаг 1: Настройка Git (рекомендуется)

```bash
# Глобальная настройка (для всех репозиториев)
git config --global core.autocrlf input

# Или только для этого репозитория
git config core.autocrlf input
```

**Что это делает:**
- `core.autocrlf input` - при checkout НЕ конвертирует LF в CRLF
- При commit автоматически конвертирует CRLF в LF

### Шаг 2: Пере-checkout файлов (если уже склонировали)

Если вы уже склонировали репозиторий ДО настройки:

```bash
# Удалить закешированные файлы (не удаляет реальные файлы!)
git rm --cached -r .

# Восстановить файлы с правильными line endings
git reset --hard
```

### Шаг 3: Проверка настройки

```bash
# Проверить настройку autocrlf
git config core.autocrlf

# Должно вывести: input
```

## Jupyter Notebook на Windows

### Проблема

Даже с правильными настройками git, Jupyter может **пересохранить** `.ipynb` файл с CRLF при:
- Запуске ячеек
- Сохранении (Ctrl+S)
- Auto-save

### Решение: Pre-commit Hook

Наш pre-commit hook **автоматически исправит** это при коммите:

```bash
git add notebooks/titanic/titanic_analysis.ipynb
git commit -m "Update notebook"

# Hook запустится автоматически:
# Checking Jupyter notebooks for line ending issues...
#   Fixing CRLF → LF: notebooks/titanic/titanic_analysis.ipynb
# ✓ Fixed line endings in 1 file(s)
```

### Альтернатива: nbstripout (опционально)

Для дополнительной очистки metadata можно установить `nbstripout`:

```bash
pip install nbstripout

# Настройка для репозитория
nbstripout --install

# Это удалит из коммитов:
# - execution_count
# - outputs (опционально)
# - metadata (опционально)
```

**Минус:** Теряется вывод ячеек в git (output cells).

## Проверка текущего файла

Проверить, какие line endings используются:

### Bash/Git Bash:
```bash
file notebooks/titanic/titanic_analysis.ipynb
# Должно быть: "JSON text data" (БЕЗ "with CRLF line terminators")
```

### Python:
```python
with open('notebooks/titanic/titanic_analysis.ipynb', 'rb') as f:
    content = f.read()

crlf = content.count(b'\r\n')
lf = content.count(b'\n') - crlf

print(f"CRLF: {crlf}, LF: {lf}")
# Правильно: CRLF: 0, LF: >0
```

## Редакторы кода

### VS Code

В `.vscode/settings.json` (создать если нет):

```json
{
  "files.eol": "\n",
  "jupyter.notebook.lineNumbers": true
}
```

### PyCharm

Settings → Editor → Code Style:
- Line separator: Unix and macOS (\n)

## FAQ

**Q: Нужно ли что-то делать каждый раз?**
A: Нет! После первоначальной настройки git и hook всё работает автоматически.

**Q: Можно ли отключить hook?**
A: Да, просто удалите `.git/hooks/pre-commit`

**Q: Hook не работает!**
A: Проверьте:
```bash
ls -la .git/hooks/pre-commit
# Должно быть: -rwxr-xr-x (исполняемый)

# Если нет, сделайте исполняемым:
chmod +x .git/hooks/pre-commit
```

**Q: Почему не использовать `core.autocrlf true`?**
A: `autocrlf true` конвертирует LF→CRLF при checkout, что сломает многие инструменты на Windows (Python, bash скрипты, Jupyter).

## Рекомендуемая конфигурация

**Оптимальная настройка для Windows:**

```bash
# Git настройка
git config --global core.autocrlf input
git config --global core.eol lf

# Для новых репозиториев
git config --global init.defaultBranch main
```

---

**Итог:** С этими настройками вы можете спокойно работать с Jupyter на Windows, и git будет всегда хранить файлы с правильными LF line endings.
