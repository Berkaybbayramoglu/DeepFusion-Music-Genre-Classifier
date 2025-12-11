# Music Genre Classification (Multi-modal)

Bu depo, çoklu modalite (mel-spektrogram, MFCC, Chroma, Tempogram) kullanarak müzik türü sınıflandırması için bir PyTorch eğitim betiği içerir.

Özellikler
- Tek dosya ile eğitim yürütülebilir: `Propes_model`.
- CLI argümanları ile özellik dosyası, epoch sayısı, batch size ve öğrenme oranı değiştirilebilir.
- Model eğitimi sırasında en iyi model kaydedilir ve eğitim geçmişi CSV olarak yazılır.

Hızlı Başlangıç

1. Ortamı hazırlayın (önerilen):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Özellik dosyanızı hazırlayın veya yolunu belirtin. Varsayılan yol: `/kaggle/input/morefeatures2improveddataset/gtzan_multi_modal_features_hps2.pkl`.

3. Eğitimi başlatın:

```bash
# Tercih edilen: paketlenmiş runner
python train.py --features /path/to/features.pkl --epochs 100 --batch-size 64 --lr 1e-4 --model-dir ./models

# Alternatif (eski wrapper uyumluluğu):
python Propes_model --features /path/to/features.pkl --epochs 100
```

Çıktılar
- `--model-dir` içinde en iyi model `best_multi_modal_gtzan_model.pth` olarak kaydedilir.
- `training_results_single_split.csv` eğitim geçmişini içerir.
- `training_history_single_split.png` eğitim/validasyon eğrilerini içerir.

Notlar
- Bu proje, `torchvision` içindeki `resnet50` ön eğitimli ağı kullanır; tek kanallı (grayscale) mel-spektrogram girişine uyarlanmıştır.
- Özelliklerin (mel_spec_orig, mel_spec_harm, mel_spec_perc, mfcc, chroma, tempogram) olduğu bir pickle dosyası beklenir.

Katkıda Bulunma
- İyileştirmeler için pull request gönderin. Lütfen yeni bağımlılıklar ekliyorsanız `requirements.txt`'i güncelleyin.

Lisans
- Lisans eklemek isterseniz bir `LICENSE` dosyası ekleyin.
