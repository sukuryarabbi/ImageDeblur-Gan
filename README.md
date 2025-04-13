# ImageDeblur-Gan

## Proje Tanımı

**ImageDeblur-Gan** projesi, bulanık fotoğrafları netleştirmeyi amaçlayan bir GAN (Generative Adversarial Network) modelidir. Bu model, bulanık görüntüleri daha net hale getirebilmek için eğitilmiş bir yapay zeka sistemini kullanır. Proje, fotoğraf restorasyonu ve görüntü iyileştirme alanlarında kullanılabilir.

## Kod Yapısı

Bu proje, aşağıdaki bileşenlerden oluşmaktadır:

1. **DatasetLoader**  
   Mevcut veri kümesindeki görüntüleri alır ve bunları iki sınıfa ayırır: bulanık (blurred) ve net (sharp). Bu, modelin eğitimi için veri hazırlığı sağlar.

2. **VGG16**  
   Daha önceden eğitilmiş olan VGG-16 modelini kullanır. Bu model, nesne tespiti ve özellik çıkarımı yaparak, insan gözünün bulanıklığı algılayıp giderme noktasında daha etkili bir şekilde çalışmasını sağlar.

3. **Generator**  
   GAN modelinin üretici kısmıdır. Bu sınıf, bulanık fotoğrafları alır ve net fotoğraflar üretir. Modelin amacı, bulanık görüntülerden mümkün olan en net görüntüyü oluşturmak ve iyileştirme yapmaktır.

4. **Discriminator**  
   Generator tarafından üretilen netleştirilmiş fotoğrafları, orijinal (net) fotoğraflarla karşılaştırır ve bu karşılaştırmalar sonucunda modelin performansını geri besler. Bu, modelin gerçekçi ve net fotoğraflar üretmesini sağlar.

5. **Train**  
   Modelin eğitim döngüsünün bulunduğu ana bileşendir. Burada eğitim parametreleri belirlenir ve modelin öğrenmesi sağlanır. Eğitim sırasında **Generator** ve **Discriminator** karşılıklı olarak eğitilir.

6. **Test**  
   Modelin test edileceği ve doğrulama yapılacağı bölüm. Eğitim sonrası elde edilen model burada test edilir ve doğruluk oranları ölçülür.

## Notlar

- Kod şu an test aşamasındadır ve performans beklentisinin altında kalmaktadır. Bulanık fotoğrafları düzeltebilmekte, ancak beklenen sonuçlar henüz tam anlamıyla elde edilememiştir.
- Projeye ilerleyen dönemlerde daha fazla iyileştirme ve test aşamaları eklenecektir.
