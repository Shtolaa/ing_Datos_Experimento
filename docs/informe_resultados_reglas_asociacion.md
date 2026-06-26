# Informe de resultados: Reglas de asociación sobre actividad en redes sociales

## Resumen ejecutivo

El experimento aplicó reglas de asociación al dataset **Social Media User Activity Dataset**, usando `self_reported_happiness` como variable de interés. El objetivo fue identificar combinaciones frecuentes de hábitos de uso, interacción digital y estilo de vida que se asocian con distintos niveles de felicidad autorreportada.

Los resultados muestran dos patrones principales:

- La felicidad alta (`self_reported_happiness__8_to_10`) se asocia con menor intensidad de uso, sesiones más moderadas y menor estrés percibido.
- La felicidad baja (`self_reported_happiness__0.999_to_3`) se asocia con mayor estrés, mayor interacción y mayor tiempo de uso en la aplicación.

FP-Growth y Apriori obtuvieron exactamente los mismos resultados bajo la configuración usada. Eclat generó menos reglas porque se ejecutó con una muestra separada y más pequeña para evitar problemas de memoria.

## Configuración del experimento

El flujo aplicado fue:

1. Carga del CSV local.
2. Normalización de nombres de columnas.
3. Análisis de cardinalidad y eliminación de columnas no adecuadas.
4. Selección de columnas de análisis definidas en el módulo del proyecto.
5. Limpieza de valores faltantes.
6. Normalización de categorías.
7. Muestreo de trabajo.
8. Agrupación de categorías raras.
9. Discretización de variables numéricas por cuantiles.
10. Creación de matriz binaria sparse.
11. Filtrado de items con soporte menor a `MIN_SUPPORT`.
12. Ejecución secuencial de FP-Growth, Apriori y Eclat.
13. Generación y análisis de reglas relacionadas con felicidad.

Parámetros principales usados en la ejecución local:

| Parámetro | Valor |
|---|---:|
| Muestra para FP-Growth y Apriori | 100,000 filas |
| Muestra para Eclat | 20,000 filas |
| `MAX_BINS` | 5 |
| `MIN_SUPPORT` | 0.02 |
| `MIN_CONFIDENCE` | 0.40 |
| `MIN_LIFT` | 1.00 |
| `MAX_ITEMSET_LENGTH` | 3 |
| `ECLAT_MAX_COMBINATION` | 2 |

## Comparación de algoritmos

| Algoritmo | Itemsets frecuentes | Reglas totales | Reglas con felicidad | Soporte promedio | Confianza promedio | Lift promedio | Lift máximo |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP-Growth | 57,676 | 41,550 | 2,427 | 0.0344 | 0.5874 | 1.7680 | 4.6390 |
| Apriori | 57,676 | 41,550 | 2,427 | 0.0344 | 0.5874 | 1.7680 | 4.6390 |
| Eclat | 10,371 | 849 | 26 | 0.0989 | 0.5111 | 1.2349 | 1.6964 |

FP-Growth y Apriori produjeron el mismo número de itemsets, reglas y reglas relacionadas con felicidad. Sus métricas también son idénticas, por lo que ambos algoritmos encontraron el mismo conjunto de patrones bajo los umbrales definidos.

Eclat produjo menos reglas porque no se ejecutó sobre la misma matriz completa. Esta decisión fue necesaria porque `pyECLAT` convierte la matriz binaria en transacciones con strings, lo que incrementa mucho el consumo de memoria.

Para continuar el análisis, el algoritmo más conveniente es **FP-Growth**, porque ofrece los mismos resultados que Apriori y suele ser más eficiente en memoria y tiempo para datasets grandes.

## Lectura de las métricas

Las reglas se interpretan con tres métricas principales:

- **Soporte:** proporción de registros donde aparece la combinación completa de la regla.
- **Confianza:** proporción de casos donde aparece el consecuente dado que ya aparece el antecedente.
- **Lift:** fuerza de asociación entre antecedente y consecuente frente a lo esperado por azar. Un lift mayor que 1 indica asociación positiva.

Ejemplo: si una regla tiene confianza de 0.48 hacia felicidad alta, significa que el 48% de los registros que cumplen el antecedente también presentan felicidad alta. Si además el lift es 2.45, esa felicidad alta aparece 2.45 veces más de lo esperado respecto a su frecuencia base.

## Distribución base de felicidad relevante

En las reglas donde la felicidad aparece como único consecuente, los soportes base observados fueron:

| Nivel de felicidad | Soporte base |
|---|---:|
| Felicidad baja `0.999_to_3` | 0.2994 |
| Felicidad alta `8_to_10` | 0.1987 |

Esto significa que, en la muestra usada, la felicidad baja aparece aproximadamente en el 29.94% de los registros y la felicidad alta en el 19.87%.

## Patrones fuertes encontrados

### Perfil asociado con felicidad alta

Las reglas más fuertes relacionadas con felicidad alta muestran usuarios con menor estrés y menor intensidad de uso. En particular, aparecen repetidamente estas condiciones:

- Estrés percibido bajo o moderado.
- Menor tiempo diario en feed.
- Menor tiempo diario en mensajes.
- Menor tiempo diario en reels.
- Menor cantidad de likes dados.
- Sesiones más cortas o menos frecuentes.

Las reglas donde la felicidad alta aparece como consecuente muestran que ciertos patrones duplican o más la presencia esperada de felicidad alta.

| Antecedente | Consecuente | Soporte | Confianza | Lift |
|---|---|---:|---:|---:|
| `average_session_length_minutes__5_to_12` AND `sessions_per_day__0.999_to_4` | `self_reported_happiness__8_to_10` | 0.0345 | 0.4875 | 2.4530 |
| `perceived_stress_score__8_to_16` AND `time_on_feed_per_day__2_to_41` | `self_reported_happiness__8_to_10` | 0.0244 | 0.4653 | 2.3414 |
| `likes_given_per_day__11_to_66` AND `perceived_stress_score__8_to_16` | `self_reported_happiness__8_to_10` | 0.0242 | 0.4597 | 2.3133 |
| `average_session_length_minutes__5_to_12` AND `time_on_feed_per_day__2_to_41` | `self_reported_happiness__8_to_10` | 0.0385 | 0.4551 | 2.2901 |
| `average_session_length_minutes__5_to_12` AND `time_on_messages_per_day__0.999_to_13` | `self_reported_happiness__8_to_10` | 0.0381 | 0.4537 | 2.2828 |

Interpretación: la felicidad alta tiene una frecuencia base de 19.87%, pero en estos subgrupos sube aproximadamente a 45% - 49%. Esto indica una asociación relevante entre uso moderado, menor estrés y felicidad alta.

### Perfil asociado con felicidad baja

Las reglas relacionadas con felicidad baja muestran un patrón opuesto. Aparecen con frecuencia condiciones de mayor interacción e intensidad de uso:

- Mayor estrés percibido.
- Muchos likes dados por día.
- Muchos comentarios escritos por día.
- Muchos mensajes directos enviados por semana.
- Mayor tiempo diario en feed.
- Mayor tiempo diario en reels, mensajes o explore.

Las reglas donde la felicidad baja aparece como consecuente muestran asociaciones consistentes con actividad digital intensa.

| Antecedente | Consecuente | Soporte | Confianza | Lift |
|---|---|---:|---:|---:|
| `likes_given_per_day__171_to_317` AND `perceived_stress_score__24_to_32` | `self_reported_happiness__0.999_to_3` | 0.0338 | 0.5946 | 1.9862 |
| `perceived_stress_score__24_to_32` AND `time_on_feed_per_day__145_to_321` | `self_reported_happiness__0.999_to_3` | 0.0337 | 0.5743 | 1.9185 |
| `comments_written_per_day__50_to_80` AND `dms_sent_per_week__40_to_80` | `self_reported_happiness__0.999_to_3` | 0.0664 | 0.5471 | 1.8276 |
| `comments_written_per_day__50_to_80` AND `perceived_stress_score__24_to_32` | `self_reported_happiness__0.999_to_3` | 0.0313 | 0.5471 | 1.8274 |
| `dms_sent_per_week__40_to_80` AND `time_on_feed_per_day__145_to_321` | `self_reported_happiness__0.999_to_3` | 0.0700 | 0.5449 | 1.8202 |

Interpretación: la felicidad baja tiene una frecuencia base de 29.94%, pero en estos subgrupos sube aproximadamente a 54% - 59%. Esto refuerza la asociación entre estrés, alta interacción y felicidad baja.

## Reglas destacadas del punto 12

El punto 12 del notebook muestra reglas donde `self_reported_happiness` puede aparecer tanto en el antecedente como en el consecuente. Cuando aparece en el antecedente, la regla no predice felicidad, sino que describe qué otros comportamientos suelen aparecer junto con ese nivel de felicidad.

### Ejemplos con felicidad alta como parte del antecedente

| Regla | Soporte | Confianza | Lift |
|---|---:|---:|---:|
| `perceived_stress_score__-0.001_to_8` AND `self_reported_happiness__8_to_10` -> `time_on_feed_per_day__2_to_41` | 0.0413 | 0.9476 | 4.6390 |
| `perceived_stress_score__-0.001_to_8` AND `self_reported_happiness__8_to_10` -> `time_on_messages_per_day__0.999_to_13` | 0.0404 | 0.9281 | 4.6084 |
| `perceived_stress_score__-0.001_to_8` AND `self_reported_happiness__8_to_10` -> `likes_given_per_day__11_to_66` | 0.0410 | 0.9419 | 4.6016 |

Interpretación: dentro del grupo de usuarios con bajo estrés y felicidad alta, es muy frecuente observar bajo tiempo en feed, bajo tiempo en mensajes y menor cantidad de likes dados. Estas reglas tienen confianza superior al 92%, por lo que describen un perfil muy consistente.

### Ejemplos con felicidad baja como parte del antecedente

| Regla | Soporte | Confianza | Lift |
|---|---:|---:|---:|
| `perceived_stress_score__32_to_40` AND `self_reported_happiness__0.999_to_3` -> `likes_given_per_day__171_to_317` | 0.0533 | 0.9085 | 4.6284 |
| `perceived_stress_score__32_to_40` AND `self_reported_happiness__0.999_to_3` -> `time_on_feed_per_day__145_to_321` | 0.0520 | 0.8865 | 4.4720 |
| `perceived_stress_score__32_to_40` AND `self_reported_happiness__0.999_to_3` -> `comments_written_per_day__50_to_80` | 0.0495 | 0.8452 | 4.3843 |

Interpretación: dentro del grupo con alto estrés y felicidad baja, aparecen fuertemente conductas de interacción alta: muchos likes, mucho tiempo en feed y muchos comentarios. Estas reglas también tienen confianza muy alta, entre 84% y 91%.

## Resultados de Eclat

Eclat encontró menos reglas debido a la muestra reducida y al límite de combinaciones. Aun así, sus reglas principales confirman el patrón de felicidad baja asociado con mayor intensidad de uso.

| Antecedente | Consecuente | Soporte | Confianza | Lift |
|---|---|---:|---:|---:|
| `comments_written_per_day__50_to_80` | `self_reported_happiness__0.999_to_3` | 0.0952 | 0.5014 | 1.6964 |
| `likes_given_per_day__171_to_317` | `self_reported_happiness__0.999_to_3` | 0.0977 | 0.5010 | 1.6949 |
| `time_on_feed_per_day__145_to_321` | `self_reported_happiness__0.999_to_3` | 0.0980 | 0.4991 | 1.6885 |
| `dms_sent_per_week__40_to_80` | `self_reported_happiness__0.999_to_3` | 0.0882 | 0.4865 | 1.6458 |
| `time_on_reels_per_day__87_to_195` | `self_reported_happiness__0.999_to_3` | 0.0927 | 0.4819 | 1.6304 |

Estos resultados son útiles como confirmación parcial, pero no deben compararse de forma directa con FP-Growth y Apriori porque Eclat no usó exactamente el mismo volumen de datos.

## Interpretación general

Los patrones encontrados son consistentes con la hipótesis del proyecto: existen combinaciones frecuentes de hábitos de uso e interacción digital asociadas con distintos niveles de felicidad autorreportada.

En términos generales:

- Los usuarios con felicidad alta tienden a aparecer junto con menor estrés y menor intensidad de actividad en la aplicación.
- Los usuarios con felicidad baja tienden a aparecer junto con mayor estrés, mayor tiempo de uso y mayor volumen de interacciones.
- Las variables de comportamiento digital más recurrentes en las reglas relevantes son `likes_given_per_day`, `time_on_feed_per_day`, `comments_written_per_day`, `dms_sent_per_week`, `time_on_messages_per_day` y `time_on_reels_per_day`.
- `perceived_stress_score` aparece como una variable clave para diferenciar perfiles de felicidad alta y baja.

## Limitaciones

Las reglas de asociación no prueban causalidad. Por ejemplo, no se puede concluir que usar más el feed cause menor felicidad, ni que menor estrés cause felicidad alta. Solo se puede afirmar que esos patrones aparecen juntos con frecuencia mayor a la esperada.

Además, las variables numéricas fueron discretizadas en intervalos. Esto permite aplicar reglas de asociación, pero la interpretación depende de los cortes generados por cuantiles.

Eclat fue limitado por memoria y se ejecutó con una muestra menor. Por esa razón, sus resultados deben considerarse complementarios.

## Conclusión

El experimento logró identificar reglas interpretables que relacionan hábitos de uso de redes sociales con niveles de felicidad autorreportada.

El mejor algoritmo para continuar el análisis es **FP-Growth**, ya que obtuvo las mismas reglas que Apriori con el mismo lift máximo y el mismo número de reglas relacionadas con felicidad, pero es más adecuado para trabajar con datasets grandes.

La conclusión principal es que la felicidad alta se asocia con uso más moderado y menor estrés, mientras que la felicidad baja se asocia con mayor estrés y mayor intensidad de interacción digital. Estos hallazgos cumplen el objetivo del proyecto al producir reglas claras, cuantificables e interpretables mediante soporte, confianza y lift.
