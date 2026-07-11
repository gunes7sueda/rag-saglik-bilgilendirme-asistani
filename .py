from google.colab import drive
drive.mount('/content/drive')

!pip install -q google-generativeai chromadb \
    langchain-text-splitters langchain \
    beautifulsoup4 requests pypdf tiktoken lxml

import os, time, requests
from pathlib import Path
from typing import List, Dict, Optional

try:
    from google.colab import userdata
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
    print('✅ GEMINI_API_KEY Secrets\'ten alındı')
except Exception:
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY','')
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = input('Gemini API anahtarınızı girin: ').strip()

import google.generativeai as genai
genai.configure(api_key=GEMINI_API_KEY)

from bs4 import BeautifulSoup
from pypdf import PdfReader
import chromadb
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

EMBED_MODEL = 'models/gemini-embedding-001'
CHAT_MODEL  = 'models/gemini-flash-latest'

chat_model = genai.GenerativeModel(CHAT_MODEL)
print(f'✅ Embedding : {EMBED_MODEL}')
print(f'✅ Chat      : {CHAT_MODEL}')
print('✅ Tüm kütüphaneler hazır')

URLS = [
    'https://www.who.int/tr/news-room/fact-sheets/detail/diarrhoeal-disease',
    'https://www.who.int/tr/news-room/fact-sheets/detail/malnutrition',
]

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def scrape_url(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        for t in soup(['script','style','nav','footer','header','aside','iframe']):
            t.decompose()
        main = None
        for sel in ['article','main','[role="main"]','.content','#content']:
            main = soup.select_one(sel)
            if main: break
        if not main: main = soup.find('body')
        text = '\n'.join(l.strip() for l in main.get_text('\n').splitlines() if l.strip())
        h1 = soup.find('h1')
        title = h1.get_text(strip=True) if h1 else url.split('/')[-1]
        print(f'  ✅ {title[:60]} ({len(text)} kr)')
        return {'url':url,'title':title,'text':text,'source':'web'}
    except Exception as e:
        print(f'  ❌ {url}: {e}')
        return None

def extract_pdf(path):
    try:
        r = PdfReader(path)
        text = '\n\n'.join(f'[S{i+1}]\n{p.extract_text()}' for i,p in enumerate(r.pages) if p.extract_text())
        name = Path(path).stem
        print(f'  ✅ PDF: {name} ({len(r.pages)} sayfa)')
        return {'url':path,'title':name,'text':text,'source':'pdf'}
    except Exception as e:
        print(f'  ❌ PDF: {e}')
        return None

print('🌐 Web scraping...')
scraped = [d for u in URLS if (d:=scrape_url(u))]

PDF_PATHS = []
pdf_docs  = [d for p in PDF_PATHS if (d:=extract_pdf(p))]

all_docs = scraped + pdf_docs

if not all_docs:
    print('⚠️  URL\'ler açılmadı — demo verisi yükleniyor...')
    all_docs = [
        {'url':'demo://ishal','title':'İshal Hastalığı','source':'demo','text':(
            'İshal Hastalığı - WHO\n'
            'İshal, günde 3+ kez sulu dışkılama olarak tanımlanır.\n'
            'Her yıl 5 yaş altı 525.000 çocuk ishalden hayatını kaybeder.\n'
            'Nedenler: Rotavirüs, E.coli, Salmonella, kirli su.\n'
            'Tedavi: ORS (Oral Rehidrasyon Solusyonu) ile sıvı replasmanı.\n'
            'Çinko takviyesi 10-14 gün verilmeli.\n'
            'Antibiyotik sadece bakteriyel ishailde, doktor önerisiyle.\n'
            'Korunma: el yıkama, güvenli su, rotavirüs aşısı, anne sütü.')
        },
        {'url':'demo://demir','title':'Demir Eksikliği Anemisi','source':'demo','text':(
            'Demir Eksikliği Anemisi\n'
            'Dünya genelinde en yaygın beslenme bozukluğudur.\n'
            'Çocukların %40, gebe kadınların %50sinde görülür.\n'
            'Preparatlar: ferröz sülfat, ferröz glukonat, ferrik formlar.\n'
            'Çocuk dozu: 3-6 mg/kg/gün elemental demir.\n'
            'Yetişkin dozu: 150-200 mg/gün, 2-3 bölünmüş doz.\n'
            'Gebe dozu: profilaksi 30-60 mg/gün, tedavi 120-200 mg/gün.\n'
            'C vitamini ile aç karnına alınırsa emilim artar.\n'
            'Çay, kahve, sütle birlikte alınmamalı. Tedavi 3-6 ay.\n'
            'Yan etkiler: bulantı, kabızlık, dışkı kararması.')
        },
        {'url':'demo://malnut','title':'Çocuk Beslenmesi Malnütrisyon','source':'demo','text':(
            'Çocuk Beslenmesi ve Malnütrisyon\n'
            'Akut malnütrisyon (wasting): boya göre düşük ağırlık.\n'
            'F75 ve F100 terapötik mama ile tedavi edilir.\n'
            'RUTF (Kullanıma Hazır Terapötik Mama) ev tedavisinde kullanılır.\n'
            'Kronik malnütrisyon (stunting): yaşa göre boy kısalığı.\n'
            'İlk 1000 gün (gebelik+2yıl) en kritik dönemdir.\n'
            'A vitamini: 6-59 ay arası çocuklara 6 ayda bir yüksek doz.\n'
            'Çinko: ishal tedavisinde ve büyüme desteğinde verilir.\n'
            'Demir: anemi taraması sonrası gerektiğinde başlanır.')
        }
    ]
    print(f'✅ {len(all_docs)} demo doküman hazır')

print(f'\n📚 Toplam doküman: {len(all_docs)}')

enc = tiktoken.get_encoding('cl100k_base')
tok = lambda t: len(enc.encode(t))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=80,
    length_function=tok,
    separators=['\n\n\n','\n\n','\n','. ',' ','']
)

def create_chunks(docs):
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc['text'])
        for i, text in enumerate(splits):
            if tok(text) < 30: continue
            chunks.append({
                'id': f"{doc['title'][:20].replace(' ','_')}_{i}",
                'text': text,
                'metadata': {
                    'source': doc['url'], 'title': doc['title'],
                    'chunk_index': i, 'token_count': tok(text),
                    'doc_type': doc.get('source','web')
                }
            })
        print(f'  📄 "{doc["title"][:40]}" -> {len(splits)} chunk')
    return chunks

print('✂️ Chunking...')
all_chunks = create_chunks(all_docs)
toks = [c['metadata']['token_count'] for c in all_chunks]
print(f'\n✅ {len(all_chunks)} chunk | Ort: {sum(toks)//len(toks)} token | Min: {min(toks)} | Max: {max(toks)}')

CHROMA_PATH = './chroma_db'
COLLECTION  = 'saglik_rag'

def get_embeddings(texts: List[str], batch: int = 20) -> List[List[float]]:
    """Gemini text-embedding-004 ile batch embedding."""
    out = []
    for i in range(0, len(texts), batch):
        b = texts[i:i+batch]
        result = genai.embed_content(
            model='models/gemini-embedding-001',
            content=b,
            task_type='retrieval_document'
        )
        out.extend(result['embedding'] if len(b)==1 else result['embedding'])
        print(f'   {min(i+batch, len(texts))}/{len(texts)} embedding tamam')
        time.sleep(1)
    return out

chroma = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    chroma.delete_collection(COLLECTION)
except Exception:
    pass
col = chroma.create_collection(COLLECTION, metadata={'hnsw:space':'cosine'})
print(f'✅ ChromaDB hazır: "{COLLECTION}"')

print(f'\n🔢 {len(all_chunks)} chunk için Gemini embedding...')
texts = [c['text'] for c in all_chunks]
embeddings = get_embeddings(texts)

BATCH = 50
for i in range(0, len(all_chunks), BATCH):
    b = all_chunks[i:i+BATCH]
    col.add(
        ids=[c['id'] for c in b],
        embeddings=embeddings[i:i+BATCH],
        documents=[c['text'] for c in b],
        metadatas=[c['metadata'] for c in b]
    )

print(f'\n✅ ChromaDB\'de {col.count()} chunk kayıtlı!')

SYSTEM = """Sen Türkçe konuşan, sağlık alanında uzmanlaşmış bir AI asistanısın.\nSana verilen BAGLAM BILGISI'ni kullanarak soruları yanıtla.\nKurallar:\n- YALNIZCA verilen bağlama dayan, kendi bilgini kullanma\n- Bağlamda yoksa: 'Bu konuda elimdeki belgede bilgi bulunamadı' de\n- Yanıt sonunda kullandığın kaynak başlıklarını listele\n- Her zaman Türkçe yanıt ver"""

def rag_query(question: str, n: int = 4, history=None) -> Dict:
    q_emb = genai.embed_content(
        model=EMBED_MODEL,
        content=question,
        task_type='retrieval_query'
    )['embedding']

    res   = col.query(query_embeddings=[q_emb], n_results=n,
                      include=['documents','metadatas','distances'])
    docs  = res['documents'][0]
    metas = res['metadatas'][0]
    dists = res['distances'][0]

    ctx = '\n\n---\n\n'.join(
        f'[Kaynak {i+1}: {m["title"]} | benzerlik: {1-d:.2f}]\n{doc}'
        for i,(doc,m,d) in enumerate(zip(docs,metas,dists))
    )
    sources = [{'title':m['title'],'url':m['source'],'sim':round(1-d,3)}
               for m,d in zip(metas,dists)]

    prompt = f"{SYSTEM}\n\nBAGLAM BILGISI:\n{ctx}\n\nSORU: {question}"

    if history:
        chat = chat_model.start_chat(history=history)
        resp = chat.send_message(prompt)
    else:
        resp = chat_model.generate_content(prompt)

    return {
        'question': question,
        'answer':   resp.text,
        'sources':  sources
    }

print('✅ RAG pipeline hazır!')

def show(r):
    print('='*65)
    print(f'❓ {r["question"]}')
    print('-'*65)
    print(r['answer'])
    print('-'*65)
    seen = set()
    for s in r['sources']:
        if s['title'] not in seen:
            print(f'  📚 {s["title"]} (benzerlik: {s["sim"]})')
            seen.add(s['title'])
    print()

SORULAR = [
    'İshalde ne yapılmalı? Hangi ilaçlar kullanılır?',
    'Demir ilacı nasıl ve ne kadar dozda kullanılmalı?',
    'Çocuklarda malnütrisyon nasıl tedavi edilir?',
]

print('🧪 Test soruları çalışıyor...\n')
for q in SORULAR:
    show(rag_query(q))
    time.sleep(2)

def chat():
    print('💬 Sağlık RAG Asistanı (Gemini)')
    print('   Çıkış: quit  |  Sıfırla: reset')
    print('='*55)
    while True:
        try:
            q = input('\n🙋 Soru: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGörüşürüz!'); break
        if not q: continue
        if q.lower() in ['quit','exit','çıkış','q']:
            print('Görüşürüz!'); break
        r = rag_query(q)
        print(f'\n🤖 {r["answer"]}')
        seen = set()
        for s in r['sources']:
            if s['title'] not in seen:
                print(f'   📚 {s["title"]}')
                seen.add(s['title'])
        time.sleep(1)


from collections import Counter

def db_stats():
    metas = col.get(include=['metadatas'])['metadatas']
    counts = Counter(m['title'] for m in metas)
    print(f'📊 Toplam: {col.count()} chunk')
    for title, n in counts.items():
        print(f'   • {title[:50]}: {n} chunk')

def add_document(text, title, source_url='manual'):
    chunks = create_chunks([{'url':source_url,'title':title,'text':text,'source':'manual'}])
    embs = get_embeddings([c['text'] for c in chunks])
    col.add(ids=[c['id'] for c in chunks], embeddings=embs,
            documents=[c['text'] for c in chunks],
            metadatas=[c['metadata'] for c in chunks])
    print(f'✅ "{title}" eklendi ({len(chunks)} chunk). Toplam: {col.count()}')

db_stats()