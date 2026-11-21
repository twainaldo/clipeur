// Clipeur pour X : pose un bouton « ciseaux » à droite de la barre d'actions de chaque tweet.
// Un clic envoie le lien du tweet à Clipeur, qui télécharge la vidéo et garde le texte comme hook.

(() => {
  const SVG = 'http://www.w3.org/2000/svg';
  const TITRES = {
    repos: 'Envoyer ce tweet à Clipeur',
    envoi: 'Envoi à Clipeur…',
    file: 'Dans la file de Clipeur…',
    pret: 'Prêt dans Clipeur',
  };
  // X redessine les tweets quand on fait défiler : l'état de chacun est gardé ici, par numéro de tweet.
  const memoire = new Map();

  function icone() {
    const svg = document.createElementNS(SVG, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    const forme = (nom, attributs) => {
      const e = document.createElementNS(SVG, nom);
      for (const k in attributs) e.setAttribute(k, attributs[k]);
      svg.append(e);
    };
    forme('circle', { cx: 6, cy: 6, r: 3 });
    forme('circle', { cx: 6, cy: 18, r: 3 });
    forme('line', { x1: 20, y1: 4, x2: 8.12, y2: 15.88 });
    forme('line', { x1: 14.47, y1: 14.48, x2: 20, y2: 20 });
    forme('line', { x1: 8.12, y1: 8.12, x2: 12, y2: 12 });
    return svg;
  }

  // Le lien d'un tweet est celui qui porte sa date.
  function lienDe(article) {
    const date = article.querySelector('a[href*="/status/"] time');
    const a = date && date.closest('a');
    const m = a && a.href.match(/^https:\/\/(?:x|twitter)\.com\/[^/]+\/status\/\d+/);
    return m ? m[0] : '';
  }

  function peindre(b) {
    const [nom, titre] = memoire.get(b.dataset.tweet) || ['repos', TITRES.repos];
    b.dataset.etat = nom;
    b.title = titre;
    b.setAttribute('aria-label', titre);
  }

  function noter(id, nom, titre) {
    memoire.set(id, [nom, titre || TITRES[nom] || '']);
    document.querySelectorAll('.clipeur-bouton[data-tweet="' + id + '"]').forEach(peindre);
  }

  function parler(message, suite) {
    try {
      chrome.runtime.sendMessage(message, r => suite(chrome.runtime.lastError ? null : r));
    } catch (e) {
      suite(null); // l'extension a été rechargée : il faut recharger la page de X
    }
  }

  function suivre(id, essai) {
    if (essai > 90) return;
    setTimeout(() => parler({ type: 'clipeur-etat', id }, r => {
      if (!r || !r.ok) return;
      if (r.etat === 'pret') noter(id, 'pret');
      else if (r.etat === 'erreur') noter(id, 'rate', r.erreur || 'Clipeur n\'a pas pu prendre ce tweet.');
      else suivre(id, essai + 1);
    }), 2000);
  }

  function envoyer(id, lien) {
    const encours = (memoire.get(id) || [])[0];
    if (encours === 'envoi' || encours === 'file') return;
    noter(id, 'envoi');
    parler({ type: 'clipeur-envoyer', lien }, r => {
      if (!r) { noter(id, 'rate', 'Recharge cette page de X, puis reclique.'); return; }
      if (!r.ok) { noter(id, 'rate', r.erreur || 'Clipeur a refusé ce tweet.'); return; }
      noter(id, 'file');
      suivre(id, 0);
    });
  }

  function poser(article) {
    const barre = article.querySelector('div[role="group"][aria-label]') || article.querySelector('div[role="group"]');
    const lien = lienDe(article);
    if (!barre || !lien || barre.querySelector('.clipeur-bouton')) return;
    const id = lien.split('/status/')[1];
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'clipeur-bouton';
    b.dataset.tweet = id;
    b.append(icone());
    peindre(b);
    b.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); envoyer(id, lien); });
    barre.append(b);
  }

  let prevu = false;
  function voir() {
    prevu = false;
    document.querySelectorAll('article[data-testid="tweet"]').forEach(poser);
  }
  new MutationObserver(() => { if (!prevu) { prevu = true; setTimeout(voir, 250); } })
    .observe(document.documentElement, { childList: true, subtree: true });
  voir();
})();
