# Proyecto Final. Machine Learning Trading System

## Abtract

Este proyecto fue elaborado por **Esteban Ortiz Tirado, Andrea Jiménez Orozco y Paloma Martinez**, como un trabajo final para la materia de **Microestructura y Sistemas de Trading**, la cual es parte del curriculum de la licenciatura en **Ingeniería Financiera**, ofertada por la universidad **ITESO**. En el presente trabajo se plantea la creación de un sistema de trading automatizado cuya generación de señales se basa en un modelo de aprendizaje de máquina. 

## Objetivos

La intención del presente trabajo es proponer, crear y analizar un sistema de trading que utiliza *Machine Learning* para la generación señales de compra/venta para contratos de **BTCUSD** operados en **MetaTrade5** plataforma de trading. Así como evaluar su *desempeño* y *efectividad* a lo largo de un determinado horizonte de tiempo por medio de un *backtest*.

En este proyecto se emplean técnicas avanzadas de optimización para el entrenamiento del modelo, el cual sirve para la construcción de cuatro criterios básicos al momento de hacer trading. Tales como:

1. Uso de datos
2. Generación de señales
3. Dimensionamiento de la posiciones
4. Toma de pérdidas y ganancias

### Para facilitar la lectura, comentarios y conclusiones descargar y/o consultar el archivo de notebook.py/notebook.html 

## Estructura de proyecto

La estructura de proyecto se basa en crear diferentes archivos de Python los cuales mandamos llamar al inicio en una sola línea dentro del cuaderno de Jupyter. Cada función tiene un objetivo distinto, consecutivo y complementario para la creación del análisis. Esto nos permite obtener como resultado una organización más limpia, funcional y eficiente, lo que se traduce en una mejor comprensión para el lector.

Los elementos son los siguientes:

- **main.py**
  *contiene el orden secuencial de las funciones principales para correr el análisis.*
- **functions.py**
  *contiene todas las funciones referentes al cálculo de ambas estrategias.*
- **data.py**
  *contiene todas las funciones referentes a la descarga y manejo de datos.*
- **visualizations.py**
  *contiene todas las funciones correspondientes a la creación de gráficas y/o elementos visuales.*

## Instalar dependencias

Instalar todas las dependencias que se encuentran en el archivo **requirements.txt**, solo corra el siguiente comando en su terminal:

        pip install -r requirements.txt
   
## Licencia
**GNU General Public License v3.0**

Los permisos de esta licencia están condicionados a poner a disposición el código fuente completo de los trabajos con licencia y las modificaciones, que incluyen trabajos más grandes que utilizan un trabajo con licencia, bajo la misma licencia. Se deben conservar los avisos de derechos de autor y licencias. Los contribuyentes proporcionan una concesión expresa de derechos de patente.

## Contacto

¿Qué es Ingeniería Financiera? En Ingeniería Financiera aprendes a formular estrategias comerciales y proyectos de inversión que puedan asegurar la viabilidad y rentabilidad de los planes de generación de riqueza de empresas, gobiernos y particulares. Desarrollas competencias para construir modelos matemáticos de diferentes escenarios de negocio considerando los objetivos de los inversores y los riesgos que puedan surgir, permitiendo así que empresas y comunidades se transformen en espacios de bienestar y desarrollo. Aplicas los conocimientos adquiridos a problemas como el desarrollo de nuevos productos financieros, propones estrategias para estimular el crecimiento de las empresas, y tomas decisiones basadas en la ciencia de datos y la simulación de escenarios de negocio que convierten los riesgos en oportunidades de crecimiento y competitividad.

¿Qué es ITESO? ITESO es la Universidad Jesuita de Guadalajara. Fundada en 1957, pertenece a una red de más de 228 universidades jesuitas de todo el mundo. Todos comparten una tradición de 450 años de educación jesuita, una tradición que históricamente ha estado en el centro del pensamiento mundial, conocida por educar líderes en todos los campos de la ciencia y el arte. El ITESO es conocido por su excelencia académica, una profunda preocupación por los contextos tanto locales como globales, y su compromiso con la mejora de las condiciones de vida de las personas. Su proyecto educativo integral busca desarrollar la inteligencia y la sensibilidad, formar jóvenes libres y socialmente responsables de por vida, en un entorno ideal para el descubrimiento y el crecimiento. https://www.iteso.mx/

Para tener mas información acerca de este repositorio y su funcionalidad favor de ponerse en contacto al siguiente correo: if706970@iteso.mx, if708924@iteso.mx, if707429@iteso.mx
