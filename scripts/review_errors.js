(function (root) {
  'use strict';
  const limbs = ['arm_left', 'arm_right', 'leg_left', 'leg_right'];

  function matchingResults(record, filters) {
    if (filters.split && filters.split !== 'all' && record.split !== filters.split) return [];
    if (filters.casualty && filters.casualty !== 'all' && record.casualty !== filters.casualty) return [];
    const search = (filters.search || '').trim().toLowerCase();
    if (search && !`${record.id} ${record.filename} ${record.casualty} ${record.filepath}`.toLowerCase().includes(search)) return [];
    if (filters.issue === 'flagged' && !record.issues.length) return [];
    if (filters.issue === 'likely_mismatch' && !record.issues.includes('likely_mismatch')) return [];
    if (filters.issue === 'uncertain' && !record.issues.includes('uncertain')) return [];
    if (filters.issue === 'crop' && !record.cropCaution) return [];
    return record.results.filter(r => {
      const [model, limb, truth, prediction] = r;
      if (filters.model === 'learned' && model === 'majority') return false;
      if (filters.model && !['all', 'learned'].includes(filters.model) && model !== filters.model) return false;
      if (filters.limb && filters.limb !== 'all' && limb !== filters.limb) return false;
      if (filters.truth && filters.truth !== 'all' && truth !== filters.truth) return false;
      if (filters.prediction && filters.prediction !== 'all' && prediction !== filters.prediction) return false;
      return !filters.errors || truth !== prediction;
    });
  }

  function selectRecords(records, filters) {
    return records.map(record => ({record, matches: matchingResults(record, filters)})).filter(x => x.matches.length);
  }

  function csvCell(value) {
    return `"${String(value == null ? '' : value).replaceAll('"', '""')}"`;
  }

  const api = {matchingResults, selectRecords, csvCell};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  root.ErrorReview = api;
  if (typeof document === 'undefined' || !document.getElementById('review-data')) return;

  const data = JSON.parse(document.getElementById('review-data').textContent);
  const byId = new Map(data.records.map(r => [String(r.id), r]));
  const tableRows = new Map([...document.querySelectorAll('tr[data-record]')].map(row => [row.dataset.record, row]));
  const tbody = document.getElementById('file-rows');
  const ids = ['split', 'model', 'limb', 'truth', 'prediction', 'casualty', 'issue', 'search', 'sort', 'perpage'];
  const controls = Object.fromEntries(ids.map(id => [id, document.getElementById(id)]));
  const errorsControl = document.getElementById('errors');
  let page = 0;
  let current = [];
  const params = new URLSearchParams(location.hash.slice(1));
  ids.forEach(id => { if (params.has(id)) controls[id].value = params.get(id); });
  errorsControl.checked = params.get('errors') !== '0';
  let hashTimer;

  function refresh(resetPage = true) {
    if (resetPage) page = 0;
    const filters = Object.fromEntries(ids.map(id => [id, controls[id].value]));
    filters.errors = errorsControl.checked;
    current = selectRecords(data.records, filters);
    const wrongScore = x => Math.max(-1, ...x.matches.filter(r => r[2] !== r[3] && r[4] != null).map(r => r[4]));
    current.sort((a, b) => {
      if (filters.sort === 'source') return a.record.id - b.record.id;
      if (filters.sort === 'confidence') return wrongScore(b) - wrongScore(a) || a.record.id - b.record.id;
      if (filters.sort === 'magnification') return b.record.gain - a.record.gain || a.record.id - b.record.id;
      return b.matches.filter(r => r[2] !== r[3]).length - a.matches.filter(r => r[2] !== r[3]).length || a.record.id - b.record.id;
    });
    const perPage = filters.perpage === 'all' ? Math.max(current.length, 1) : Number(filters.perpage || 24);
    const pages = Math.max(1, Math.ceil(current.length / perPage));
    page = Math.min(page, pages - 1);
    tableRows.forEach(row => { row.style.display = 'none'; });
    const fragment = document.createDocumentFragment();
    current.slice(page * perPage, (page + 1) * perPage).forEach(({record, matches}) => {
      const row = tableRows.get(String(record.id));
      row.style.display = 'table-row';
      const wrong = matches.filter(r => r[2] !== r[3]);
      const target = row.querySelector('.matching-results');
      const shown = (wrong.length ? wrong : matches).slice(0, 3);
      target.textContent = shown.map(r => `${data.modelNames[r[0]]}: ${r[1]} ${r[2]} → ${r[3]}`).join('\n');
      if (matches.length > 3) target.textContent += `\n… ${matches.length} matching model/limb results`;
      fragment.appendChild(row);
    });
    tbody.appendChild(fragment);
    const wrongDecisions = current.reduce((sum, x) => sum + x.matches.filter(r => r[2] !== r[3]).length, 0);
    document.getElementById('match-count').textContent = `${current.length} images · ${wrongDecisions} mismatched model/limb decisions`;
    document.getElementById('page-count').textContent = current.length ? `Showing ${page * perPage + 1}–${Math.min((page + 1) * perPage, current.length)} · page ${page + 1}/${pages}` : 'No matching images';
    document.getElementById('previous').disabled = page === 0;
    document.getElementById('next').disabled = page >= pages - 1;
    document.getElementById('empty').hidden = current.length !== 0;
    clearTimeout(hashTimer);
    hashTimer = setTimeout(() => {
      const p = new URLSearchParams();
      ids.forEach(id => { if (controls[id].value) p.set(id, controls[id].value); });
      p.set('errors', errorsControl.checked ? '1' : '0');
      try { history.replaceState(null, '', `${location.pathname}${location.search}#${p.toString()}`); } catch (_) { /* Filters still work without URL updates. */ }
    }, 200);
  }

  ids.forEach(id => controls[id].addEventListener(id === 'search' ? 'input' : 'change', () => refresh()));
  errorsControl.addEventListener('change', () => refresh());
  document.getElementById('previous').addEventListener('click', () => { page--; refresh(false); });
  document.getElementById('next').addEventListener('click', () => { page++; refresh(false); });
  document.getElementById('reset').addEventListener('click', () => {
    ids.forEach(id => { controls[id].value = id === 'split' ? document.body.dataset.defaultSplit : id === 'search' ? '' : id === 'sort' ? 'errors' : id === 'perpage' ? '24' : 'all'; });
    errorsControl.checked = true; refresh();
  });
  document.getElementById('export-filtered').addEventListener('click', () => {
    const rows = [['source_record','filename','split','casualty_id','model','limb','supplied_label','predicted_label','mismatch','predicted_score','case_page']];
    current.forEach(({record, matches}) => matches.forEach(r => rows.push([record.id,record.filename,record.split,record.casualty,data.modelNames[r[0]],r[1],r[2],r[3],r[2] !== r[3],r[4],`cases/source_${String(record.id).padStart(6, '0')}.html`])));
    const url = URL.createObjectURL(new Blob([rows.map(row => row.map(csvCell).join(',')).join('\r\n')], {type:'text/csv;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = 'filtered_model_results.csv'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.getElementById('interactive').hidden = false;
  refresh();
})(typeof globalThis === 'undefined' ? this : globalThis);
