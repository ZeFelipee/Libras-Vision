// Service worker: permite instalar o app e usar offline depois da primeira abertura.
const CACHE = "libras-v1";

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(["./", "index.html", "modelo_libras.json", "manifest.json", "icon-192.png", "icon-512.png"])));
  self.skipWaiting();
});

self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(nomes => Promise.all(nomes.filter(n => n !== CACHE).map(n => caches.delete(n)))));
  self.clients.claim();
});

self.addEventListener("fetch", e => {
  if (e.request.method !== "GET") return;
  const mesmaOrigem = new URL(e.request.url).origin === location.origin;
  if (mesmaOrigem) {
    // arquivos do app: tenta a rede (pega modelo novo) e cai no cache se estiver offline
    e.respondWith(fetch(e.request).then(r => { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); return r; }).catch(() => caches.match(e.request)));
  } else {
    // MediaPipe, modelo da mão e fontes: cache primeiro
    e.respondWith(caches.match(e.request).then(h => h || fetch(e.request).then(r => { const c = r.clone(); caches.open(CACHE).then(x => x.put(e.request, c)); return r; })));
  }
});
