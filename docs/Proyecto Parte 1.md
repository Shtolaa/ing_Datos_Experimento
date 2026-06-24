# **PARTE 1: Definición del Problema, Dataset y EDA**

## **Enfoque:** 

### **1\. Ficha Técnica y Descripción General del Dataset**

* **Nombre del Dataset:** Social Media User Activity Dataset (Exploring Social Media Usage Patterns).  
* **Fuente del Dataset:** Repositorio público de investigación de ciencia de datos (Dataset tabular sintético a gran escala diseñado para modelamiento predictivo y análisis de comportamiento digital). Es un dataset real y aprobado para fines académicos.  
* **Licencia / Condiciones de Uso:** Dominio Público / Creative Commons (CC0: Public Domain), lo que permite su manipulación, entrenamiento de modelos y exposición pública sin restricciones comerciales ni de derechos de autor.  
* **Descripción General:** El conjunto de datos recopila información detallada sobre el comportamiento, interacciones digitales, datos demográficos y hábitos de salud física de usuarios de redes sociales. Permite cruzar el estilo de vida de las personas con sus métricas exactas de uso de la aplicación.  
* **Volumen de Datos:** El dataset es altamente robusto, contando con **1,500,000 de registros** (filas) y más de 50 columnas, lo que garantiza una representatividad estadística ideal para algoritmos clásicos de Machine Learning.  
* **Variables Identificadas y Tipos:**  
  * **Demográficas/Socioeconómicas (Categóricas y Numéricas):** age, gender, country, urban\_rural, income\_level, employment\_status, education\_level, relationship\_status, has\_children.  
  * **Estilo de Vida y Salud (Numéricas y Categóricas):** exercise\_hours\_per\_week, sleep\_hours\_per\_night, diet\_quality, smoking, perceived\_stress\_score, self\_reported\_happiness, body\_mass\_index, daily\_steps\_count, weekly\_work\_hours.  
  * **Métricas de Interacción Digital (Numéricas):** sessions\_per\_day, average\_session\_length\_minutes, posts\_created\_per\_week, likes\_given\_per\_day, comments\_written\_per\_day, dms\_sent\_per\_week, followers\_count, following\_count.  
  * **Distribución del Tiempo en la App (Numéricas en minutos):** time\_on\_feed\_per\_day, time\_on\_explore\_per\_day, time\_on\_messages\_per\_day, time\_on\_reels\_per\_day.  
  * **Preferencias y Configuración (Categóricas):** content\_type\_preference, preferred\_content\_theme, privacy\_setting\_level.

### **2\. Definición del Problema de Machine Learning**

* **Contexto:** Actualmente existe interés en comprender cómo ciertos hábitos de uso de redes sociales se relacionan con el bienestar percibido por los usuarios. Sin embargo, analizar variables individuales no siempre permite identificar combinaciones de factores que aparecen junto con distintos niveles de felicidad. Por ello, se propone aplicar reglas de asociación para descubrir patrones frecuentes entre actividad, interacción y felicidad autorreportada. 

* **Tipo de Tarea de Machine Learning:** Reglas de asociación.

* **Principal interés:** 

`self_reported_happiness`

* **Pregunta / Hipótesis de Guía:** ¿Qué combinaciones de hábitos de uso e interacción en redes sociales se asocian frecuentemente con niveles altos, medios o bajos de `self_reported_happiness`? 

## **Preparación inicial del dataset**

Para aplicar reglas de asociación sin perder la granularidad de las variables numéricas, no se transformarán inicialmente todas las variables continuas en rangos amplios como “bajo”, “medio” o “alto”. En cambio, se mantendrán los valores numéricos originales durante el análisis exploratorio para conservar la precisión de las métricas de comportamiento digital, salud y estilo de vida.

Esta decisión se justifica porque variables como `average_session_length_minutes`, `time_on_reels_per_day`, `sleep_hours_per_night`, `exercise_hours_per_week`, `perceived_stress_score` y `self_reported_happiness` pueden perder información relevante si se agrupan demasiado pronto. Por ejemplo, usuarios con 2 y 6 horas de uso podrían quedar en una misma categoría, aunque representen comportamientos distintos.

En la etapa inicial, se trabajará con las variables numéricas en su formato original para analizar distribuciones, correlaciones, valores extremos y posibles patrones. Luego, para la aplicación de reglas de asociación, se evaluarán métodos que permitan trabajar con atributos numéricos o generar intervalos de manera controlada, evitando una discretización arbitraria.

La variable `self_reported_happiness` seguirá siendo el eje principal del análisis, pero se priorizará conservar su valor numérico original durante el EDA. 

De esta manera, la preparación inicial del dataset buscará mantener la mayor cantidad posible de información original, evitando una pérdida prematura de granularidad y permitiendo que las decisiones de transformación se basen en evidencia del análisis exploratorio.

## **Criterio preliminar de éxito**

El proyecto se considerará exitoso si se logran obtener reglas de asociación interpretables que relacionen hábitos de uso de redes sociales, variables de interacción digital y condiciones de estilo de vida con distintos valores o niveles de `self_reported_happiness`.

Los principales criterios preliminares de éxito serán:

1. Obtener reglas con valores adecuados de soporte, confianza y lift.  
2. Identificar combinaciones frecuentes de variables asociadas a distintos comportamientos de felicidad autorreportada.  
3. Mantener la mayor granularidad posible de las variables numéricas durante el análisis inicial.  
4. Justificar cualquier transformación aplicada a variables numéricas mediante criterios estadísticos y no mediante rangos arbitrarios.  
5. Interpretar las reglas encontradas de forma clara, señalando que representan asociaciones frecuentes y no relaciones causales.

