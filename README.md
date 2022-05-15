# DL-Fleuret

In questa repo metteremo il nostro lavoro per il corso di DL con Francois.

Luca  **suca!**

## Struttura

I progetti sono contenuti nelle cartelle **Miniproject_1** e **Miniproject_2.**
- **Miniproject_1:**
    - **model.py** contiene la NN definita nella classe ```Model()``` con tutti i suoi metodi. 
    Per importarla: 
        ```python
        from Miniproject_1.model import Model
        ```
- **Network_MK1.ipynb** è un notebook da usare per il **testing.**

## Da Fare
- **weight decay**: plot errors (train e validation) in funzione di indice batch* indice epochs
- provare **step size** e momentum
- implementare ```load_pretrained_model```
- trovare il modo di castare l'output a **INT**
- Sistemare divisione per 255