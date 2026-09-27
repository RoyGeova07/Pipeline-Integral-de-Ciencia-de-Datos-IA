# Pipeline Integral de Ciencia de Datos & IA — Aprobación de Préstamos

**Asignatura:** Ciencia de Datos / Inteligencia Artificial
**Integrantes:** [Roy Umaña](https://github.com/RoyGeova07), [Jose Martin](https://github.com/jmartinrivera11), [Joseph Valerio](https://github.com/JackJack64-TURBO)
**Dataset:** [Loan Approval Dataset (Kaggle)](https://www.kaggle.com/datasets/rohitgrewal/loan-approval-dataset?resource=download)

---

## Contexto del proyecto

Este proyecto desarrolla un pipeline completo de ciencia de datos aplicado a una institución financiera, cubriendo desde la gobernanza de datos hasta la comunicación ética de resultados. El objetivo es predecir si una solicitud de préstamo será **aprobada o rechazada** a partir de las características financieras y demográficas del solicitante, y traducir el desempeño del modelo en un impacto financiero concreto para el negocio.

## Problema a resolver

- **Tipo de problema:** Clasificación binaria.
- **Variable objetivo:** `loan_status` (Approved / Rejected).
- **Variables predictoras clave:** puntaje crediticio (CIBIL), ingresos anuales, monto y plazo del préstamo, valor de activos residenciales, comerciales, de lujo y bancarios, nivel educativo, situación laboral (independiente o no) y número de dependientes.

El reto de negocio consiste en apoyar al área de crédito/riesgo a tomar decisiones más consistentes sobre las solicitudes, minimizando el costo financiero de los errores de clasificación (aprobar a alguien que debía ser rechazado, o rechazar a alguien que debía ser aprobado).

## Estructura del pipeline (5 fases)

1. **Formulación del problema y gobernanza de datos** — Definición del problema de negocio, roles de Data Owner y Data Steward, y estrategia de privacidad (enmascaramiento del identificador `loan_id`).
2. **Auditoría de calidad y análisis exploratorio (EDA)** — Revisión de nulos y duplicados, estadísticas descriptivas, segmentación por educación/empleo, y visualizaciones diagnósticas (boxplots, distribución de variables clave).
3. **Modelado estadístico y capacidad de generalización** — Comparación de dos modelos competidores (Regresión Logística vs. Árbol de Decisión), validación cruzada 5-Fold, y diagnóstico de sobreajuste (Train vs. Test).
4. **Traducción a métricas financieras y de negocio** — Matriz de confusión, reporte de clasificación, y asignación de costos monetarios a Falsos Positivos y Falsos Negativos para estimar el ahorro del modelo frente a escenarios extremos (aprobar todo / rechazar todo).
5. **Comunicación ética, limitaciones y defensa** — Declaración de sesgos en los datos de origen y límites operativos del modelo.

## Herramientas y tecnologías utilizadas

- **Python** (Jupyter Notebook)
- **pandas / numpy** — manipulación y análisis de datos
- **matplotlib / seaborn** — visualización
- **scikit-learn** — modelado (`LogisticRegression`, `DecisionTreeClassifier`, `Pipeline`, `StandardScaler`, `StratifiedKFold`, `cross_val_score`, métricas de clasificación)

## Qué se resolvió / resultados principales

- **Regresión Logística:** Accuracy de 91.45% (train) y 92.27% (test); CV 5-Fold promedio de 91.25% (± 1.37%). Sin señales claras de sobreajuste.
- **Árbol de Decisión (profundidad máxima 5):** Accuracy de 97.28% (train) y 97.66% (test); CV 5-Fold promedio de 96.57% (± 0.60%). Mejor desempeño general y variación reducida entre particiones.
- **Impacto financiero:** usando el Árbol de Decisión como modelo final y costos estimados de 5,000 por Falso Positivo y 1,000 por Falso Negativo, el costo total de los errores del modelo fue de 32,000, frente a 1,615,000 si se aprobaran todas las solicitudes y 531,000 si se rechazaran todas. Esto representa un ahorro estimado de **1,583,000** frente a aprobar todo y de **499,000** frente a rechazar todo.
- **Limitaciones identificadas:** sub-representación de perfiles de bajos ingresos (ingreso mínimo de 200,000, promedio de 5.05M) y 28 observaciones con activos residenciales negativos; el modelo no debe usarse para solicitantes sin `cibil_score`, con ingresos menores a 200,000, plazos mayores a 20 años o montos superiores a 39.5M.

## Estructura del repositorio

```
├── Plantilla_Proyecto.ipynb   # Notebook con las 5 fases del pipeline
├── loan_approval_dataset.csv  # Dataset original (Kaggle)
└── README.md
```

## Cómo ejecutar

1. Instalar dependencias: `pip install pandas numpy matplotlib seaborn scikit-learn`
2. Abrir `Plantilla_Proyecto.ipynb` en Jupyter Notebook/Lab.
3. Ejecutar las celdas en orden; el notebook lee `loan_approval_dataset.csv` desde el mismo directorio.
