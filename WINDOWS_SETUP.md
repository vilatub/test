# Настройка для Windows

Этот документ описывает настройку репозитория для работы на Windows, особенно при работе с Jupyter notebooks.

## Проблема: Line Endings (CRLF vs LF)

**Проблема:** Jupyter на Windows сохраняет файлы `.ipynb` с Windows line endings (`\r\n` = CRLF), что создает проблемы:
- Диффы в git выглядят сломанными (показывают изменения во всех строках)
- Текстовые редакторы показывают `^M` символы
- Проблемы при работе в команде (Windows + Linux/Mac)

**Решение:** Принудительное использование Unix line endings (LF) для всех текстовых файлов.

## Что уже настроено

### 1. `.gitattributes` (Приоритет выше, чем core.autocrlf)

Файл `.gitattributes` автоматически конвертирует line endings при checkout/commit:

```gitattributes
*.ipynb text eol=lf
*.py text eol=lf
*.md text eol=lf
```

**Что это означает:**
- `text` = git знает, что это текстовый файл
- `eol=lf` = **при checkout** git нормализует в LF (переопределяет `core.autocrlf`!)

**Важно:** `.gitattributes` имеет **приоритет выше** чем `core.autocrlf`!

```
Приоритет (от высшего к низшему):
1. .gitattributes (в репозитории)
2. core.autocrlf (в .git/config)
3. Глобальный core.autocrlf
```

Поэтому даже если у кого-то `core.autocrlf true`, `.gitattributes` заставит использовать LF.

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
- `core.autocrlf input` - при **checkout** НЕ конвертирует LF в CRLF
- При **commit** автоматически конвертирует CRLF в LF

**Означает:**
```
Git (LF) → checkout → Рабочая директория (LF)  ← Файлы остаются с LF!
         ← commit  ← Рабочая директория (CRLF) ← Jupyter сохранил с CRLF
Git (LF)                                        ← Автоматически конвертируется
```

**Почему это правильно для Windows:**
- ✅ Jupyter, VS Code, PyCharm корректно работают с LF
- ✅ Python скрипты работают с LF (Windows Python понимает оба формата)
- ✅ Git Bash, WSL работают с LF
- ✅ Node.js, большинство инструментов разработки работают с LF
- ⚠️ Только старый Notepad не понимал LF (но в Windows 10+ исправлено)

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

### Что происходит с Jupyter на Windows:

```
1. git checkout
   Git (LF) → Диск: notebook.ipynb (LF)

2. jupyter notebook
   Открывает: notebook.ipynb (LF) ✅ Читает нормально

3. Работа в Jupyter
   Запуск ячеек, редактирование...

4. Ctrl+S или Auto-save
   Сохраняет: notebook.ipynb (CRLF) ⚠️ Jupyter на Windows по умолчанию использует CRLF

5. git add notebook.ipynb
   Git staging area: (CRLF) ⚠️ Пока еще CRLF

6. git commit
   → Pre-commit hook запускается!
   → Обнаруживает CRLF
   → Конвертирует CRLF → LF
   → Re-stage файл
   → Commit продолжается

   Git (LF) ✅ В репозитории всегда LF!
```

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

A: Сравним три варианта:

| Настройка | Checkout (git → диск) | Commit (диск → git) | Проблемы |
|-----------|----------------------|---------------------|----------|
| `false` | LF → LF | Ничего | ⚠️ CRLF попадет в git |
| `input` | LF → LF | CRLF → LF | ✅ Лучший для Windows |
| `true` | LF → CRLF | CRLF → LF | ❌ Ломает Python, bash, Jupyter |

**С `autocrlf true`:**
```bash
# Файлы получают CRLF при checkout
git checkout main
# Python скрипт с CRLF
python script.py  # Может сломаться!
# Bash скрипт с CRLF
./script.sh       # Ошибка: $'\r': command not found
```

**С `autocrlf input` (наш выбор):**
```bash
# Файлы остаются с LF при checkout
git checkout main
# Python скрипт с LF
python script.py  # ✅ Работает (Python понимает LF на Windows)
# Bash скрипт с LF
./script.sh       # ✅ Работает (Git Bash понимает LF)
# Jupyter notebook с LF
jupyter notebook  # ✅ Открывает нормально
                  # Может сохранить с CRLF - но hook исправит при commit!
```

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
