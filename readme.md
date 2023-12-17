## Курс по разработке рекомендательных систем

### О курсе
- Университет: __ИТМО__
- Авторы курса: крутые специалисты из МТС

### Изучаемые темы
1. Разработка микросервиса с использованием тестов и линтеров
2. Валидация и метрики в рекомендательных системах
3. Бейзлайны (популярные айтемы)
4. Модели, основанные на kNN
5. Модели, основанные на матричной факторизации: ALS, LightFM
6. Методы приближенного поиска соседей

### Запуск
#### Сервис
```cmd
poetry run uvicorn src.app:app --port 5000
```
#### Тесты
```cmd
poetry run pytest ./tests -W ignore::DeprecationWarning
```
#### Линтеры
```cmd
poetry run pylint src
poetry run flake8 src
poetry run black --check src
```

### Домашние задания
#### Домашнее задание №1
&#9745; Написать с нуля сервис с простой моделью

&#9745; Получить токен и зарегистрировать модель в телеграмм-боте

&#9745; Добавить аутентификацию и тесты на нее

&#9745; Предусмотреть, чтобы сервис возвращал 404 ошибку, если задано неверное имя модели и покрыть тестами

&#9745; Добавить описание возможных ответов в сваггер

&#9745; Писать чистый и эффективный код, соблюдая PEP8

#### Домашнее задание №2
&#9745; Разработать класс для расчёта метрик на основе кросс-валидации

&#9745; Разработать класс для визуального анализа рекомендаций

&#9745; Провести тестирование классов в чистом воспроизводимом ноутбуке с использованием:

- Модели: RandomModel(random_state=32), PopularModel из rectools
- Метрики: 2 ранжирующие, 2 классификационные, 2 beyond-accuracy для порогов 1, 5, 10. MAP обязательно
- Сплиттер: TimeRangeSplitter из rectools, 3 фолда для кросс-валидации по неделе, исключение холодных юзеров и айтемов и просмотренных айтемов
- Визуализация рекомендаций и историй просмотров для юзеров [666262, 672861, 955527]. Для айтемов обязательно отобразить названия, жанры и количество просмотров в датасете

#### Домашнее задание №3
&#9745; Рефакторинг модели с семинара

&#9745; Провести эксперименты с userKnn моделью в ноутбуке:
- придумать, что делать с холодными пользователями в тесте (3 балла)
- сделать кол-во рекомендаций равным N, а не меньше N (3 балла)
- реализовать тюнинг гиперпараметров и сделать выводы (3 балла)

&#9745; Обернуть модель в сервис (максимум 12 баллов)
- обучение модели в ноутбуке
- сохранение артефактов модели
- загрузка артефактов при старте приложения
- запрашивание рекомендаций "на лету" из более легкой версии модели

#### Домашнее задание №4
&#9745; Создать базовый класс для выдавания популярных рекомендаций холодным пользователям и фильтрации уже просмотренных айтемов у горячих пользователей

&#9745; Побить метрику на лидерборде map@10 = 0.075 (получено в итоге 0.088) с моделью из lightfm и с использованием ANN (5 баллов)

&#9745; Реализовать тюнинг гиперпараметров для моделей из implicit, lightfm с использованием Optuna (3 балла)

&#9745; Воспользоваться методом приближенного поиска соседей для выдачи рекомендаций: Flat и HNSW из библиотеки faiss (3 балла)

&#9745; Сделать рекомендации для пользователей на основе их фичей (3 балла)

&#9745; Обернуть модель в сервис и выдавать рекомендации "на лету" (12 баллов)
