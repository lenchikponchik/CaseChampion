Тестовые фото для вкладки «Диагностика» — по одному на класс модели плюс фото «не листа».
Все снимки взяты из открытых источников, лицензия указана у каждого. В скобках — top-1 модели
wambugu71/crop_leaf_diseases_vit по `python classifier.py` (CPU, 30.10.2026).

  wheat_brown_rust.jpg     — пшеница, бурая ржавчина (сценарии 1, 2, 4)      [Wheat___Brown_Rust 1.00]
      Project-AgML/wheat_disease_detection, строка 929, фрагмент по рамке категории «2»; полевая съёмка,
      Юаньян (Хэнань, Китай), 2023. CC BY 4.0. Категории датасета обезличены (0–6), «2» сопоставлена
      с бурой ржавчиной по снимкам вручную.
  wheat_yellow_rust.jpg    — пшеница, жёлтая ржавчина                          [Rice___Leaf_Blast 0.43 — ошибка]
      Wikimedia Commons, File:Stripe_rust_on_wheat.jpg. Public domain.
      Модель не узнаёт жёлтую ржавчину на этом полевом фото (Wheat___Yellow_Rust — второе место, 0.33).
  wheat_healthy.jpg        — пшеница, здоровый лист (сценарий 6)               [Wheat___Healthy 1.00]
      DigiGreen/Crop_Disease_Images, images/00040.jpg (фото фермера, Индия). CC BY 4.0.
  corn_common_rust.jpg     — кукуруза, ржавчина                               [Corn___Common_Rust 0.98]
      Project-AgML/corn_leaf_disease_classification, image_1010.png (Аньхой, Китай; Bai, 2025,
      doi 10.6084/m9.figshare.30264037). CC BY 4.0. Перекодировано в JPEG.
  corn_gray_leaf_spot.jpg  — кукуруза, церкоспороз                            [Corn___Gray_Leaf_Spot 1.00]
      PlantDoc (Singh et al., 2020) через Project-AgML/plant_doc_classification,
      «Kelley%20Gray%20leaf%20spot.jpg». CC BY-SA 4.0. Снимки PlantDoc-кукурузы входят в обучающую выборку модели.
  corn_healthy.jpg         — кукуруза, здоровый лист                          [Corn___Healthy 1.00]
      Project-AgML/maize_disease_classification, HEALTHY_438.JPEG. CC BY 4.0. Уменьшено до 1280 px.
  potato_late_blight.jpg   — картофель, фитофтороз (сценарий 3)              [Potato___Late_Blight 1.00]
      PlantVillage (Hughes, Salathé, 2015), «RS_LB 2593.JPG». CC BY-SA 3.0.
      Лабораторный снимок из обучающей выборки модели: на полевых фото фитофтороз модель
      не распознаёт (0 из 40 в PlantDoc, см. results/README.md), поэтому для сценария 3 взят этот снимок.
  potato_early_blight.jpg  — картофель, альтернариоз                         [Potato___Early_Blight 1.00]
      PlantDoc через Project-AgML/plant_doc_classification, «alternaria (24).JPG». CC BY-SA 4.0. Уменьшено до 1280 px.
  potato_healthy.jpg       — картофель, здоровый лист                        [Potato___Healthy 1.00]
      Potato Leaf Disease Dataset in Uncontrolled Environment через
      Project-AgML/potato_leaf_disease_classification, 20230802_112947115.jpg. CC BY 4.0.
  rice_brown_spot.jpg      — рис, бурая пятнистость                          [Rice___Brown_Spot 1.00]
      Project-AgML/rice_leaf_disease_classification_bd, brown_spot(167).jpg. CC BY 4.0.
  rice_leaf_blast.jpg      — рис, пирикуляриоз                               [Rice___Leaf_Blast 1.00]
      Project-AgML/rice_leaf_disease_classification_bd, leaf_blast(36).jpg. CC BY 4.0.
  rice_healthy.jpg         — рис, здоровый лист                              [Wheat___Yellow_Rust 0.46 — ошибка]
      Project-AgML/rice_leaf_bacterial_fungal (raw), «Healthy_rice_leaf (1).jpg». CC BY 4.0.
      Ни один открытый снимок здорового риса вне обучающей выборки модель не узнала
      (0 из 40 в тесте, 0 из 157 в этом датасете). С культурой «Рис» система выдаёт конфликт (R2).
  not_a_leaf.jpg           — не лист (сценарий 5)                            [Invalid 1.00]
      Wikimedia Commons, File:American_Legion_Post_304,_Manhasset,_November_30,_2023.jpg. CC0.

Класса «гельминтоспориоз кукурузы» (corn_leaf_blight) в модели нет, хотя он есть в её карточке:
настоящий id2label содержит 13 классов, поэтому фото для него не нужно.

Проверка перед защитой:  python classifier.py
Для сценариев 1–4 и 6 вероятность top-1 должна быть не ниже 0,85 — выполнено (0.996, 1.00, 0.998).
