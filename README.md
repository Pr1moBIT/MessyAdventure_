# 📊 Modelo Predictivo de Ventas - Classic Cars

## 📝 Descripción del Proyecto

Proyecto de análisis de datos enfocado en ventas de autos clásicos utilizando una base de datos SQLite. El análisis se realiza mediante Jupyter Notebooks con consultas SQL avanzadas y visualizaciones con matplotlib, proporcionando insights estratégicos para la toma de decisiones comerciales.

## 🎯 Objetivo

Analizar patrones de ventas, identificar productos estrella, evaluar estacionalidad y generar recomendaciones basadas en datos para optimizar el inventario y estrategias de marketing.

## 🔧 Tecnologías Utilizadas

- **Python 3.x**
- **pandas** - Manipulación y análisis de datos
- **numpy** - Operaciones numéricas
- **matplotlib** - Visualizaciones
- **sqlite3** - Gestión de base de datos
- **Jupyter Notebook** - Entorno de desarrollo interactivo

## 📦 Instalación

### Requisitos Previos
- Python 3.7 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. **Clonar el repositorio**
git clone https://github.com/Pr1moBIT/MessyAdventure_.git
cd CurlyADv

2. **Instalar dependencias**
pip install pandas numpy matplotlib jupyter

3. **Iniciar Jupyter Notebook**
jupyter notebook

## 🗄️ Estructura de la Base de Datos

La base de datos `classic.db` contiene las siguientes tablas principales:

- **products** - Catálogo de productos (autos clásicos)
- **orderDetails** - Detalles de transacciones
- **orders** - Información de órdenes con fechas

### Relaciones Clave
- `products.productCode` ↔ `orderDetails.productCode`
- `orders.orderNumber` ↔ `orderDetails.orderNumber`

## 📈 Análisis Implementados

### 1. **Análisis de Productos Más Vendidos**
- Ranking por ingresos totales
- Unidades vendidas por producto
- Visualización: Consulta SQL con aggregaciones

### 2. **Participación de Mercado**
- Porcentaje de ingresos por producto
- Análisis Top 10 + Others
- Visualización: Gráfico de pastel

### 3. **Precio Promedio por Unidad**
- Identificación de productos premium
- Análisis de estrategia de precios
- Cálculo: `ingresos_totales / unidades_totales`

### 4. **Análisis de Rotación de Productos**
- Frecuencia de pedidos por producto
- Unidades promedio por pedido
- Identificación de productos estrella

### 5. **Estacionalidad de Ventas**
- Ingresos mensuales agregados
- Identificación de temporadas altas/bajas
- Visualización: Gráfico de líneas temporal

## 💡 Convenciones del Código

### Nomenclatura en Español
unidades_totales          # Total units
ingresos_totales          # Total revenue
porcentaje_participacion  # Market share percentage
precio_promedio_unidad    # Average unit price
frecuencia_pedidos        # Order frequency
ingresos_mensuales        # Monthly revenue
mes_num                   # Month number

### Patrón de Análisis
# 1. Importar librerías
import pandas as pd, numpy as np, matplotlib.pyplot as plt, sqlite3

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

### Consultas SQL
- Uso de **CTEs (WITH)** para cálculos complejos
- **ROUND()** para redondear valores monetarios a 2 decimales
- **STRFTIME()** para extraer componentes de fecha
- **NULLIF()** para evitar división por cero

## 🎯 Insights Estratégicos

### Productos Estrella 🌟
- Alta frecuencia de pedidos + Alto volumen por pedido
- **Acción**: Prioridad máxima en inventario

### Productos Populares 🔥
- Alta frecuencia + Bajo volumen por pedido
- **Acción**: Stock constante, menor cantidad por unidad

### Productos de Mayoreo 📦
- Baja frecuencia + Alto volumen por pedido
- **Acción**: Enfoque en distribuidores/B2B

### Productos a Revisar ⚠️
- Baja frecuencia + Bajo volumen por pedido
- **Acción**: Evaluar descontinuación o cambio de estrategia


**Proyecto**: [MessyAdventure_](https://github.com/Pr1moBIT/MessyAdventure_)
**Autor**: Pr1moBIT