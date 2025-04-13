# ImageDeblur-Gan

# KOD YAPISI 

1-) DatasetLoader : Mevcut datasette olan verileri çekip blur ve sharp olmak üzere 2 kısıma ayırır.
2-) VGG16 : Daha önceden eğitilmiş olan VGG-16 datasetini çeker ve nesne tespiti yapar. Bu sayede insan gözü için fotoğraflarda blur etkilerini gidericeği yerleri daha iyi anlar.
3-) Generator : Gan modelinin üretici sınıfı. Input olarak verisetindeki bulanık fotoğrafları alır ve yeni bir fotoğraf üretir.
4-) Discriminator : Generatordan gelen fotoğrafları bulanık olmayan fotoğraflar ile karşılaştırıp generatora geri bildirim yapılmasını sağlar.
5-) Train : Tüm kodun birleştiği yer. Burada eğitim döngüsü ve eğitim parametreleri girilir.
6-) Test : Sonuçların test edileceği yer 

# NOT : 
Kod daha test aşamasında eklenmesi gereken belli başlı kısımlar var şuanda bulanık fotoğrafları düzeltiyor ama beklenenin altında bir performans sergiliyor