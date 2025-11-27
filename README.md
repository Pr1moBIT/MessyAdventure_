# 📊 Modelo Predictivo de Ventas - Classic Cars Models

## 📝 Descripción del Proyecto

Proyecto integral de análisis de datos y machine learning enfocado en ventas de modelos a escala y maquetas de autos clásicos coleccionables. Incluye análisis exploratorio mediante consultas SQL avanzadas, visualizaciones personalizadas con matplotlib, y un modelo predictivo de regresión para proyectar ventas mensuales futuras. El proyecto utiliza datos históricos de transacciones para generar insights estratégicos y predicciones accionables.

## 🎯 Objetivos

1. **Análisis Exploratorio**: Identificar patrones de ventas, productos estrella y estacionalidad del negocio
2. **Análisis de Pareto**: Determinar qué porcentaje de productos genera el 80% de los ingresos
3. **Modelado Predictivo**: Desarrollar un modelo de regresión Ridge para predecir ventas mensuales del próximo año
4. **Generación de Insights**: Proporcionar recomendaciones basadas en datos para optimizar inventario y estrategias comerciales

## 🔧 Tecnologías y Librerías

### Core
- **Python 3.x**
- **Jupyter Notebook** - Entorno de desarrollo interactivo

### Análisis de Datos
- **pandas** - Manipulación y análisis de datos
- **numpy** - Operaciones numéricas
- **sqlite3** - Gestión de base de datos SQLite

### Visualización
- **matplotlib** - Gráficos estáticos personalizados
- **seaborn** - Visualizaciones estadísticas (heatmaps, correlaciones)

### Machine Learning
- **scikit-learn** - Modelos de regresión y preprocesamiento
  - `LinearRegression` - Modelo base
  - `Ridge` - Regresión con regularización L2 (modelo seleccionado)
  - `Lasso` - Regresión con regularización L1
  - `RandomForestRegressor` - Modelo basado en árboles
  - `GradientBoostingRegressor` - Modelo de boosting
  - `StandardScaler` - Normalización de features
  - `MinMaxScaler` - Escalado de variables
  - `train_test_split` - División de datos train/test

### Persistencia
- **pickle** - Serialización de modelos entrenados

## 📦 Instalación

### Requisitos Previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/Pr1moBIT/MessyAdventure_.git
cd MessyAdventure_
```

2. **Instalar dependencias**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. **Iniciar Jupyter Notebook**
```bash
jupyter notebook
```

4. **Abrir el notebook principal**
- Navegar a `predictive sales model.ipynb`

## 🗄️ Estructura de la Base de Datos

La base de datos `classic.db` contiene las siguientes tablas principales:

- **products** - Catálogo de modelos a escala y maquetas coleccionables (réplicas de autos clásicos)
- **orderDetails** - Detalles de transacciones de ventas
- **orders** - Información de órdenes con fechas de compra

### Relaciones Clave
- `products.productCode` ↔ `orderDetails.productCode`
- `orders.orderNumber` ↔ `orderDetails.orderNumber`

## 📈 Análisis Implementados

### Fase 1: Análisis Exploratorio

#### 1. **Análisis de Productos Más Vendidos**
- Ranking por ingresos totales y unidades vendidas
- Identificación de productos estrella del catálogo
- Visualización: Consulta SQL con aggregaciones

#### 2. **Participación de Mercado**
- Porcentaje de ingresos por producto
- Análisis Top 10 + Others
- Visualización: Gráfico de dona con paleta teal personalizada

#### 3. **Precio Promedio por Unidad**
- Identificación de productos premium vs económicos
- Análisis de estrategia de precios
- Cálculo: `ingresos_totales / unidades_totales`

#### 4. **Análisis de Rotación de Productos**
- Frecuencia de pedidos por producto
- Unidades promedio por pedido
- Segmentación: Estrella / Popular / Mayoreo / A Revisar

#### 5. **Estacionalidad de Ventas**
- Ingresos mensuales agregados (2003-2005)
- Identificación de temporadas altas (Oct-Nov) y bajas (Jun-Jul)
- Visualización: Gráfico de líneas con estilo minimalista

#### 6. **Análisis de Pareto (80/20)**
- 71 productos (65.14%) generan el 80% de los ingresos
- Total de ingresos: $9,604,190.61
- Recomendación: Optimizar inventario en productos de cola larga

### Fase 2: Preparación de Datos para Modelado

#### 7. **Ingeniería de Features**
- Agregación mensual de ventas totales
- Creación de lag features (ventas de meses anteriores)
- Media móvil de 3 meses para capturar tendencias
- Normalización con MinMaxScaler

#### 8. **Análisis de Correlación**
- Matriz de correlación entre variables
- Heatmap con seaborn (paleta coolwarm)
- Identificación de features predictivas

### Fase 3: Modelado Predictivo

#### 9. **Entrenamiento de Modelos**
Modelos evaluados:
- Linear Regression
- **Ridge Regression** ⭐ (Modelo seleccionado)
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

**Métricas de evaluación:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coeficiente de determinación)

#### 10. **Predicciones Futuras**
- Proyección de ventas mensuales para el próximo año (12 meses)
- Visualización comparativa: histórico vs predicción
- Formato de salida: Tabla resumen + gráfico temporal
- Modelo serializado con pickle para reutilización

## 💡 Convenciones del Código

### Nomenclatura en Español
```python
unidades_totales          # Total units
ingresos_totales          # Total revenue (target variable)
porcentaje_participacion  # Market share percentage
precio_promedio_unidad    # Average unit price
frecuencia_pedidos        # Order frequency
ingresos_mensuales        # Monthly revenue
mes_num / month           # Month number
year / año                # Year
año_mes                   # Year-month format (YYYY-MM)
```

### Patrón de Análisis SQL
```python
# 1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# 2. Conectar a base de datos
con = sqlite3.connect("classic.db")

# 3. Definir consulta SQL
q = """
SELECT ...
FROM ...
GROUP BY ...
ORDER BY ...
"""

# 4. Ejecutar y visualizar
df = pd.read_sql(q, con)
```

### Consultas SQL Avanzadas
- **CTEs (WITH)** para cálculos complejos y subqueries
- **ROUND()** para redondear valores monetarios a 2 decimales
- **STRFTIME()** para extraer componentes de fecha
- **NULLIF()** para evitar división por cero
- **Agregaciones**: SUM, COUNT, AVG, MIN, MAX
- **JOINs**: INNER JOIN para relacionar tablas

### Estilo de Visualización
```python
# Configuración estándar de gráficos
plt.figure(figsize=(10, 6))
plt.plot(..., color='teal', linewidth=2, markersize=8)
plt.title('Título', color='lightgray', fontsize=14, fontweight='bold')
plt.tick_params(axis='x', colors='lightgray')
plt.tick_params(axis='y', colors='lightgray')

# Ocultar bordes (spines)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_color('lightgray')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Pipeline de Machine Learning
```python
# 1. Preparar features y target
X = df.drop(columns=['ingresos_totales'])
y = df['ingresos_totales']

# 2. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# 3. Entrenar modelo
modelo = Ridge()
modelo.fit(X_train, y_train)

# 4. Predecir
y_pred = modelo.predict(X_test)

# 5. Evaluar
from sklearn.metrics import mean_absolute_error, r2_score
print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# 6. Guardar modelo
import pickle
with open('modelo_ridge.pkl', 'wb') as f:
    pickle.dump(modelo, f)
```

## 🎯 Insights Estratégicos

### Análisis de Productos

#### Productos Estrella 🌟
- Alta frecuencia de pedidos + Alto volumen por pedido
- **Acción**: Prioridad máxima en inventario, nunca desabastecer

#### Productos Populares 🔥
- Alta frecuencia + Bajo volumen por pedido
- **Acción**: Stock constante, menor cantidad por unidad

#### Productos de Mayoreo 📦
- Baja frecuencia + Alto volumen por pedido
- **Acción**: Enfoque en distribuidores/B2B, pedidos por demanda

#### Productos a Revisar ⚠️
- Baja frecuencia + Bajo volumen por pedido
- **Acción**: Evaluar descontinuación o cambio de estrategia (38 productos = 20% de ingresos)

### Estacionalidad

- **Meses Pico**: Octubre y Noviembre (temporada alta de ventas)
- **Meses Bajos**: Junio y Julio (oportunidad para campañas promocionales)
- **Estrategia**: Preparar inventario 2 meses antes de temporada alta

### Concentración de Ingresos (Pareto)

- **71 productos críticos** (65.14% del catálogo) → 80% de ingresos
- **Implicación**: Distribución más equilibrada que Pareto clásico (20/80)
- **Ventaja**: Menor riesgo de dependencia en pocos productos
- **Desafío**: Mayor complejidad en gestión de inventario

### Modelo Predictivo

- **Modelo**: Ridge Regression (regularización L2)
- **Ventajas**: Simple, interpretable, previene overfitting
- **Uso**: Proyección de ventas mensuales para planificación presupuestaria
- **Output**: Predicciones de ingresos para próximos 12 meses


## 📁 Estructura del Proyecto

```
MessyAdventure_/
├── predictive sales model.ipynb    # Notebook principal con todo el análisis
├── classic.db                       # Base de datos SQLite
├── datos_ventas_mensuales.csv       # Dataset agregado mensual
├── df_final.pkl                     # DataFrame preprocesado serializado
├── modelo_ventas_ridge.pkl          # Modelo Ridge entrenado
├── scaler_ventas.pkl                # Scaler para normalización
└── README.md                        # Documentación del proyecto
```

## 🚀 Uso del Modelo

### Cargar Modelo Entrenado
```python
import pickle
import pandas as pd

# Cargar modelo y scaler
with open('modelo_ventas_ridge.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('scaler_ventas.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Cargar datos preprocesados
df_final = pd.read_pickle('df_final.pkl')
```

### Hacer Predicciones
```python
# Preparar nuevos datos
X_nuevo = df_final[['year', 'month', 'unidades_vendidas', 'numero_ordenes', 
                     'ticket_promedio']].tail(1)

# Normalizar
X_nuevo_scaled = scaler.transform(X_nuevo)

# Predecir
prediccion = modelo.predict(X_nuevo_scaled)[0]
print(f"Ingreso predicho: ${prediccion:,.2f}")
```

## 📊 Resultados Clave

- **Productos analizados**: 109 en total
- **Período de datos**: 2003-2005
- **Meses de datos**: ~36 registros mensuales
- **Modelo seleccionado**: Ridge Regression
- **Variables predictoras**: year, month, unidades_vendidas, numero_ordenes, ticket_promedio
- **Variable objetivo**: ingresos_totales (mensuales)

## 🔮 Próximos Pasos

1. Incorporar más features externas (estacionalidad mejorada, eventos especiales)
2. Experimentar con modelos de series temporales (ARIMA, Prophet)
3. Implementar validación cruzada temporal
4. Crear dashboard interactivo con Streamlit o Dash
5. Predicciones por línea de producto (productLine)
