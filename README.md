# face_recognition

## Постановка задачи

Необходимо разработать модель по детекции, выравниванию и распознаванию лиц
____________

## Структура репозитория

- notebooks/ - ноутбуки с экспериментами
- app/ - исходники приложения

-----
### Этап 1. Детекция.
-------

На этом этапе была дообучена на датасете с лицами модель SSD300-VGG16. Понятно, что с YOLO мало что сравнится, но SSD - это моя особая персональная любовь. 

**Результаты**:

|Параметр|Значение|
|--|--|
|Среднее значение IoU на тестовых данных|0,69|
|Максимальное значение IoU на тестовых данных|0,98|
|Среднее значение MaP50 на тестовых данных|0,97|
|Максимальное значение MaP50 на тестовых данных|1,00|

**Использованные источники**

- [датасет](https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection)
- [Статья с ArXiv про YOLO](https://arxiv.org/pdf/2105.12931.pdf)
-----
### Этап 2. Поиск контрольных точек на лице, обрезка и поворот изображения.
-------

Здесь в качестве feature extractor использовалась EfficientNetB0, в которой был добавлен регрессионный слой, предсказывающий координаты опорных точек. В качетсве опорных точек из обучающего датасета были оставлены только координаты зрачков глаз.  

**Результаты**:

RMSE координат опорных точек на тестовой выборке составила около 5px.
В то же время, основная проблема связана с тем, что в обучающем датасете практически все люди на фотографиях смотрят прямо в камеру, а угол поворота головы сравнительно невелик. Это привело к тому, что на "сырых" изображениях лиц, обрезанных после детекции, в некоторых случаях качество предсказаний было достаточно низким. После определения координат зрачков глаз рассчитывался угол наклона головы и необходимый коэффициент масштабирования по расстоянию между зрачками. Изображение доворачивалось на требуемый угол и обрезалось в размер, соответствующий входу классификатора из части 3.

**Использованные источники**

- [датасет](https://www.kaggle.com/datasets/nagasai524/facial-keypoint-detection)
- [готовая библиотека на основе RetinaFace. Выглядит многообещающе](https://github.com/mantasu/face-crop-plus/tree/main)
------

### Этап 3. Распознавание лиц.
------

Сперва на датасете CelebA500 обучались модели-классификаторы с использованием Cross-Enthropy Loss. 

**Результаты:**

|**Модель**|**Число эпох**|**Accuracy на валидации**|
|--|--|--|
|ResNet18 без заморозки слоев|15|0,6|
|ResNet18 с заморозкой слоев|10|0,71|
|ResNet50 с заморозкой слоев|10|0,69|
|DenseNet121 с заморозкой слоев|15|0,75|
|**DenseNet121 с заморозкой слоев и двумя scheduler**|**30**|**0,79**|
|MobilenetV2|10|0,69|
|EfficientNet|15|0,67|

Для лучшей модели - DenseNet121 - Accuracy на тестовой выборке составила 80 %. Интересно, что примерно на 10% повысить метрику позволило периодическое увеличение learning rate в ходе обучения.

Далее, для изображений, не присутствовавших ни в одной из выборок, были рассчитаны метрики TPR@FPR. Для этого из исходной модели был удален слой классификатора, и фактически модель стала генератором эмбеддингов. Несмотря на достаточно высокое значение accuracy, TPR@FPR на неизвестных модели ранее изображениях при требуемом низком проценте ложно-положительных срабатываний достаточно невысок. 

Также были попытки обучить модель на ArcFace Loss и TripletLoss. Для обоих вариантов были реализованы циклы обучения, для Triplet Loss был создан свой датасет. генерирующий "тройки" изображений, кастомизирована модель (добавлен выход модели с предпоследнего слоя для расчета лосса), однако подбор параметров к настоящему моменту не завершен. 


|Этап |Статус |Результат|
|--|--|--|
|Обучение на CrossEnthropy Loss| ✅ | Accuracy на тесте 0,8| 
|Расчет TPR@FPR| ✅ | Метрика до 0,52|
|Обучение на ArcFace Loss|🔄| Не завершен подбор гипермараметров|
|Обучение на Triplet Loss|🔄| Реализован кастомный датасет, кастомизирована модель, не завершен подбор гиперпараметров|



**В итоге на 3 этапе было реализовано:**

- обучение модели определения лиц на Cross Enthropy Loss. Accuracy - 0,8
- расчет TPR@FPR - до 0,52
- функции для обучения на ArcFace, но не подобраны параметры обучения
- загрузчик anchor-negative-positive изображений для triplet loss, но не подобраны гиперпараметры для обучения

**Использованные источники**

- [датасет на выровненной и обрезанной части которого обучался классификатор](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [Triplet Loss](https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch)
- [ArcFace Best Practice](https://github.com/deepinsight/insightface/tree/master)
- [Пример с ArcFace и визуализацией эмбеддингов](https://www.kaggle.com/code/nanguyen/arcface-loss)
- [еще про ArcFace](https://shubham-shinde.github.io/blogs/arcface/)
- [и пример с EfficientNet](https://www.kaggle.com/code/zaccheroni/pytorch-efficientnet-arcface-training)
  
-------
### Приложение
-------

**На основе FastAPI было создано приложение, которое:**

- принимает на вход фотографию
- детектирует лица
- обрезает найденные лица
- определяет координаты зрачков глаз
- поворачивает и обрезает изображение
- генерирует эмбеддинг лица и рассчитывает "схожесть" лиц на основе косинусного расстояния для найденных эмбеддингов

**В приложении используется 3 тюнингованных и дообученных нейросети:**

- SSD300-VGG16 для детекции лиц
- EfficientNetB0 для поиска опорных точек лица
- DenseNet121 для генерации эмбеддингов лиц

[Видео с демонстрацией работы приложения](https://drive.google.com/file/d/13PsKxWEe_iHTDoX6LVeKNPN2uyytmX5Z/view?usp=drive_link)

-----
### Что нужно улучшить
----
**В дальнейшем необходимо:**

- дообучить нейросеть второго этапа на неидеально расположенных лицах, возможно, потребуется дополнительная разметка опорных точек лица
- починить ArcFaceLoss
- дообучить TripletLoss
- интегрировать в приложение веса наилучшей модели
