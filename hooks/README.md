# Git Hooks

Этот каталог содержит полезные git hooks для автоматизации задач.

## Pre-commit Hook

**Файл:** `pre-commit`

**Назначение:** Автоматически нормализует line endings в `.ipynb` файлах (CRLF → LF) перед коммитом.

### Установка

```bash
# Скопировать hook в .git/hooks/
cp hooks/pre-commit .git/hooks/pre-commit

# Сделать исполняемым (на Linux/Mac)
chmod +x .git/hooks/pre-commit
```

### На Windows (Git Bash):

```bash
# Скопировать hook
cp hooks/pre-commit .git/hooks/pre-commit

# Hook должен работать автоматически
```

### Что делает hook:

1. При каждом `git commit` проверяет staged `.ipynb` файлы
2. Если находит Windows line endings (CRLF = `\r\n`)
3. Автоматически конвертирует в Unix line endings (LF = `\n`)
4. Повторно stage файл с исправленными line endings
5. Продолжает commit

### Пример работы:

```bash
$ git add notebooks/titanic/titanic_analysis.ipynb
$ git commit -m "Update notebook"

Checking Jupyter notebooks for line ending issues...
  Fixing CRLF → LF: notebooks/titanic/titanic_analysis.ipynb
✓ Fixed line endings in 1 file(s)
  Files have been re-staged with LF line endings

[main abc1234] Update notebook
 1 file changed, 10 insertions(+), 5 deletions(-)
```

### Требования:

- Python 3 (должен быть доступен как `python3`)
- Bash (Git Bash на Windows)

### Отключение hook:

Если нужно временно отключить:

```bash
# Переименовать
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled

# Или удалить
rm .git/hooks/pre-commit
```

### Альтернатива: nbstripout

Для более агрессивной очистки Jupyter notebooks (удаление output, execution_count):

```bash
pip install nbstripout
nbstripout --install
```

**Минус:** Теряются outputs ячеек в git истории.

## Дополнительные hooks (будущие)

В будущем могут быть добавлены:
- `pre-push` - проверка перед push
- `commit-msg` - валидация commit message
- `post-merge` - действия после merge

---

**Примечание:** Файлы в `.git/hooks/` не попадают в git (по дизайну git). Поэтому hooks хранятся в этой папке и требуют ручной установки.
