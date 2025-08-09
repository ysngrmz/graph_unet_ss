
# Graph U-Net with Semi-Supervised Learning (GraphUnet-SS)

Bu proje, **Graph U-Net** mimarisi ile protein ikincil yapı tahmini yapmak üzere geliştirilmiş bir derin öğrenme sistemidir. 
Ana çalıştırılabilir dosya `results/deep_model.py` olup, `scripts` klasöründeki veri hazırlama, performans ölçümü ve özel GCN/GAT katman modüllerine bağlıdır.

---

## 📂 Proje Yapısı

```
graph_unet_ss-main/
    results/
        deep_model.py                  # Ana model çalıştırma dosyası
    scripts/
        gends.py                        # Veri üretme ve hazırlama scripti
        generate_n_window_dataset_for_fullpredict.py # Sliding window veri seti üretici
        performance_metrics.py          # Performans metrikleri
        layers_gcn/                     
            graph_attention_cnn_layer.py
            graph_cnn_layer.py
            graph_convolutional_recurrent_layer.py
            graph_ops.py
            multi_graph_attention_cnn_layer.py
            multi_graph_cnn_layer.py
```

---

## 🔧 Gereksinimler

Aşağıdaki Python kütüphanelerinin kurulu olması gerekir:

```bash
pip install numpy tensorflow keras scikit-optimize
```

TensorFlow sürümü: >= 2.x  
Keras: TensorFlow ile entegre sürüm  
Python: 3.7+ (kod içerisinde Python 3.6 uyumlu `pyc` dosyaları mevcut)

---

## 🚀 Çalıştırma Adımları

1. **Depoyu klonlayın veya zip olarak indirin**:
    ```bash
    git clone https://github.com/ysngrmz/graph_unet_ss.git
    cd graph_unet_ss
    ```

2. **Bağımlılıkları yükleyin**:
    ```bash
    pip install -r requirements.txt
    ```
    Eğer `requirements.txt` yoksa yukarıda belirtilen paketleri manuel kurun.

3. **Modeli çalıştırın**:
    ```bash
    cd results
    python deep_model.py
    ```

`deep_model.py`, `../scripts/` ve `../scripts/layers_gcn/` dizinlerini **sys.path** içine ekleyerek özel modülleri yükler.  
Eğitim sırasında veri hazırlama (`gends.py`) ve özel GNN katman tanımları (`graph_*_layer.py`) kullanılır.

---

## 📜 Dosya Açıklamaları

- **deep_model.py** → Ana eğitim ve model tanım dosyası. CNN, LSTM, GraphCNN/GAT katmanları ve Bayes optimizasyonu içerir.
- **gends.py** → Eğitim için veri üretir / hazırlar.
- **generate_n_window_dataset_for_fullpredict.py** → Kaynak verilerden "n-window" veri seti oluşturur.
- **performance_metrics.py** → F1, Accuracy gibi metrik hesaplamalarını yapar.
- **layers_gcn/** → Özel GCN ve GAT tabanlı katman tanımları:
  - `graph_attention_cnn_layer.py`
  - `graph_cnn_layer.py`
  - `graph_convolutional_recurrent_layer.py`
  - `multi_graph_attention_cnn_layer.py`
  - `multi_graph_cnn_layer.py`
  - `graph_ops.py` → Temel graph operasyonları (adjacency işlemleri vb.)

---

## 🧪 Denemeler ve Sonuçlar

Eğitim tamamlandığında sonuçlar `results/` klasöründe saklanır.  
Model hiperparametre optimizasyonu **scikit-optimize** ile yapılmaktadır (`gp_minimize`, `forest_minimize`).

---

## 📄 Lisans

Bu proje henüz bir lisans dosyası içermiyor. Lütfen lisans bilgilerini ekleyin.
