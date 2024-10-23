# Importación de bibliotecas necesarias
import numpy as np
import scipy.stats as st
import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols

# Hipótesis 1: Evaluar si la estatura promedio de los alumnos del 7mo cuatrimestre difiere de 175 cm

# Parámetros iniciales para la primera hipótesis
media_muestra = 171.94
media_referencia = 175
desviacion_muestra = 3.91
n_muestra = 30  # Tamaño de muestra

# Crear una muestra aleatoria basada en la media y desviación estándar
alturas = np.random.normal(media_muestra, desviacion_muestra, n_muestra)

# Prueba t para una muestra
t_statistica, p_valor = st.ttest_1samp(alturas, media_referencia)

# Resultados de la Hipótesis 1
print("Hipótesis 1: ¿La media de estatura difiere de 175 cm?")
print(f"Estadístico t: {t_statistica}, Valor p: {p_valor}")
if p_valor < 0.05:
    print("Conclusión: Se rechaza la hipótesis nula, la media de estatura es significativamente diferente a 175 cm.\n")
else:
    print("Conclusión: No se rechaza la hipótesis nula, no hay evidencia para afirmar que la media sea distinta.\n")


# Hipótesis 2: Evaluar si la desviación estándar es menor o igual a 10 cm

# Parámetros para la segunda hipótesis
varianza_muestra = desviacion_muestra ** 2
varianza_esperada = 10 ** 2
chi2_statistico = (n_muestra - 1) * varianza_muestra / varianza_esperada

# Prueba Chi-cuadrado para la varianza
valor_p_chi2 = st.chi2.sf(chi2_statistico, df=n_muestra-1)

# Resultados de la Hipótesis 2
print("Hipótesis 2: ¿La desviación estándar es ≤ 10 cm?")
print(f"Chi-cuadrado estadístico: {chi2_statistico}, Valor p: {valor_p_chi2}")
if valor_p_chi2 < 0.05:
    print("Conclusión: Se rechaza la hipótesis nula, la desviación estándar es menor a 10 cm.\n")
else:
    print("Conclusión: No se rechaza la hipótesis nula, no hay suficiente evidencia para afirmarlo.\n")


# Hipótesis 3: ¿Existe una diferencia en la estatura media entre hombres y mujeres?

# Datos para la tercera hipótesis
media_hombres = 171.93
media_mujeres = 169
cantidad_hombres = cantidad_mujeres = 30
desviacion_grupos = 3.91

# Generar muestras aleatorias para cada grupo
alturas_hombres = np.random.normal(media_hombres, desviacion_grupos, cantidad_hombres)
alturas_mujeres = np.random.normal(media_mujeres, desviacion_grupos, cantidad_mujeres)

# Prueba t para dos muestras independientes
t_stat_2, p_valor_2 = st.ttest_ind(alturas_hombres, alturas_mujeres)

# Resultados de la Hipótesis 3
print("Hipótesis 3: ¿Hay una diferencia significativa en la estatura entre hombres y mujeres?")
print(f"t-estadístico: {t_stat_2}, Valor p: {p_valor_2}")
if p_valor_2 < 0.05:
    print("Conclusión: Se rechaza la hipótesis nula, hay una diferencia significativa en las estaturas.\n")
else:
    print("Conclusión: No se rechaza la hipótesis nula, no hay evidencia suficiente para afirmar que difieren.\n")


# Hipótesis 4: Diferencias de retraso entre Aerolínea A y B

# Parámetros para la cuarta hipótesis
media_retraso_A = 15
media_retraso_B = 25
desv_retraso_A = 10
desv_retraso_B = 15
n_A, n_B = 100, 100

# Generar muestras de retrasos para cada aerolínea
retrasos_A = np.random.normal(media_retraso_A, desv_retraso_A, n_A)
retrasos_B = np.random.normal(media_retraso_B, desv_retraso_B, n_B)

# Prueba t para muestras independientes
t_stat_3, p_valor_3 = st.ttest_ind(retrasos_A, retrasos_B)

# Resultados de la Hipótesis 4
print("Hipótesis 4: ¿Existen diferencias significativas en los retrasos entre Aerolínea A y B?")
print(f"t-estadístico: {t_stat_3}, Valor p: {p_valor_3}")
if p_valor_3 < 0.05:
    print("Conclusión: Se rechaza la hipótesis nula, hay diferencias significativas en los retrasos.\n")
else:
    print("Conclusión: No se rechaza la hipótesis nula, no hay evidencia de diferencias significativas en los retrasos.\n")


# Hipótesis 5: ¿El retraso promedio varía entre los días de la semana?

# Datos para la quinta hipótesis
datos = pd.DataFrame({
    'dia_semana': ['Lunes']*200 + ['Martes']*200 + ['Miércoles']*200 + ['Jueves']*200 + ['Viernes']*200 + ['Sábado']*200 + ['Domingo']*200,
    'tiempo_retraso': np.random.normal(20, 15, 200).tolist() + np.random.normal(15, 10, 200).tolist() +
                     np.random.normal(25, 20, 200).tolist() + np.random.normal(10, 5, 200).tolist() +
                     np.random.normal(30, 25, 200).tolist() + np.random.normal(5, 2, 200).tolist() +
                     np.random.normal(12, 7, 200).tolist()
})

# ANOVA de un factor para comparar entre días
modelo_anova = ols('tiempo_retraso ~ dia_semana', data=datos).fit()
resultado_anova = anova_lm(modelo_anova)

# Resultados de la Hipótesis 5
print("Hipótesis 5: ¿El retraso promedio varía entre los días de la semana?")
print(resultado_anova)
if resultado_anova["PR(>F)"].iloc[0] < 0.05:
    print("Conclusión: Se rechaza la hipótesis nula, el retraso promedio varía significativamente según el día.\n")
else:
    print("Conclusión: No se rechaza la hipótesis nula, no hay diferencias significativas entre días.\n")


# Hipótesis 6: ¿Existe correlación entre retrasos de salida y llegada?

# Datos para la sexta hipótesis
retrasos_salida = np.random.normal(20, 5, 150)
retrasos_llegada = np.random.normal(25, 7, 150)

# Prueba de correlación de Pearson
coef_corr, p_val_corr = st.pearsonr(retrasos_salida, retrasos_llegada)

# Resultados de la Hipótesis 6
print("Hipótesis 6: ¿Existe una correlación significativa entre retraso de salida y llegada?")
print(f"Coeficiente de correlación: {coef_corr}, Valor p: {p_val_corr}")
if p_val_corr < 0.05:
    print("Conclusión: Se rechaza la hipótesis nula, existe una correlación significativa entre los retrasos.\n")
else:
    print("Conclusión: No se rechaza la hipótesis nula, no hay evidencia de correlación significativa.\n")
