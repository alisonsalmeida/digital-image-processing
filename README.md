# Processamento de Imagens com OpenCV

Este repositório contém exercícios práticos de processamento de imagens utilizando a biblioteca **OpenCV** em Python. Os scripts abordam técnicas básicas e intermediárias, como segmentação, filtros, operações morfológicas e transformações de imagem.

## 📂 Estrutura dos Exercícios

### Segmentação
- Segmentação por diferença absoluta entre imagens.
- Segmentação com limiarização manual e com algoritmo de Otsu.

### Operações com Imagens
- Obtenção da imagem negativa.
- Conversão para tons de cinza com pesos específicos (luminosidade perceptiva).
- Extração e visualização de canais de cor (R, G, B).

### Redução de Ruídos
- Filtro para remoção de ruído "salt and pepper".

### Realce de Imagens
- Filtro High-Boost e Máscara de Nitidez.

### Visualizações
- Uso de `matplotlib` com layouts 2x2 para comparação de imagens.
- Conversão de imagens BGR → RGB para exibição correta no `matplotlib`.

## 🛠️ Tecnologias utilizadas

- Python 3.x
- OpenCV
- Matplotlib
- NumPy

## ▶️ Execução

Certifique-se de ter os pacotes necessários instalados:

```bash
poetry install
