// Clipeur pour X : la partie qui parle au site Clipeur (http://localhost:4848).
// La page de X ne peut pas appeler ce Mac directement ; c'est ce script qui le fait pour elle.

const SITE = 'http://localhost:4848';

async function demander(chemin, corps) {
  const options = corps
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(corps) }
    : {};
  const r = await fetch(SITE + chemin, options);
  const j = await r.json().catch(() => ({}));
  return Object.assign({ ok: r.ok }, j);
}

chrome.runtime.onMessage.addListener((msg, _qui, repondre) => {
  if (!msg || typeof msg.type !== 'string') return false;
  if (msg.type === 'clipeur-envoyer') {
    demander('/api/x', { liens: String(msg.lien || '') })
      .then(r => repondre(r.ok ? r : { ok: false, erreur: r.erreur || 'Clipeur a refusé ce tweet.' }))
      .catch(() => repondre({ ok: false, erreur: 'Clipeur ne répond pas. Lance-le (lancer.command), puis reclique.' }));
    return true; // la réponse arrive plus tard
  }
  if (msg.type === 'clipeur-etat') {
    demander('/api/x?id=' + encodeURIComponent(String(msg.id || '')))
      .then(repondre)
      .catch(() => repondre({ ok: false }));
    return true;
  }
  return false;
});

// Un clic sur l'icône de l'extension ouvre le site.
chrome.action.onClicked.addListener(() => { chrome.tabs.create({ url: SITE }); });
