# 🩺 RAG Tabanlı Sağlık Bilgilendirme Asistanı Geliştirilmesi

Bu proje, karmaşık medikal dökümanları analiz ederek kullanıcı sorgularına en doğru, güvenilir ve semantik yanıtları üreten Yapay Zeka destekli bir **RAG (Retrieval-Augmented Generation)** asistanıdır. Büyük Dil Modellerinin (LLM) halüsinasyon görme riskini azaltarak tamamen beslendiği döküman kaynaklı yanıtlar üretir.

## 🚀 Mimari ve Teknik Detaylar

* **Veri Parçalama (Data Chunking):** Yoğun medikal dökümanlar, anlam bütünlüğü korunacak şekilde akıllı parçalara (chunks) ayrılmıştır[cite: 1].
* **Vektör İndeksleme:** Parçalanan veriler, semantik arama yapılabilmesi adına vektör uzayına aktarılmış ve yüksek performanslı bir **Vektör Veritabanı** üzerinde indekslenmiştir[cite: 1].
* **Bilgi Geri Çağırma (Retrieval):** Kullanıcı bir soru sorduğunda, sistem sorgunun anlamını analiz eder ve vektör veritabanından en ilgili kaynak metinleri yakalar[cite: 1].
* **Yapay Zeka Yanıt Üretimi (LLM):** Yakalanan doğru tıbbi kaynaklar ve kullanıcı sorusu bir araya getirilerek LLM'e iletilir ve kullanıcıya doğrulanabilir bir bilgilendirme sunulur[cite: 1].

## 🛠️ Kullanılan Teknolojiler ve Araçlar

* **Dil:** Python[cite: 1]
* **Yapay Zeka Mimarisi:** RAG Sistemi, LLM Entegrasyonu[cite: 1]
* **Doğal Dil İşleme (NLP):** Veri Chunking, Semantik Arama, Metin İşleme[cite: 1]
* **Veri Depolama:** Vektör Veritabanı (Vector Database)[cite: 1]

## ⚠️ Önemli Not / Sorumluluk Reddi
Bu proje tamamen akademik ve bilgi edinme amaçlı bir RAG mimarisi çalışmasıdır. Üretilen yanıtlar kesin bir tıbbi tavsiye niteliği taşımamaktadır.
