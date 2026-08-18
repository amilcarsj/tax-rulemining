(() => {
  const filterRoot = document.querySelector('[data-pattern-filter]');
  if (!filterRoot) return;

  const search = filterRoot.querySelector('[data-pattern-search]');
  const node = filterRoot.querySelector('[data-node-filter]');
  const coverage = filterRoot.querySelector('[data-coverage-filter]');
  const output = filterRoot.querySelector('[data-coverage-output]');
  const rows = [...document.querySelectorAll('[data-pattern-row]')];
  const emptyMessage = document.querySelector('.empty-filter-message');

  function applyFilters() {
    const searchText = search.value.trim().toLowerCase();
    const selectedNode = node.value;
    const minimumCoverage = Number(coverage.value);
    let visibleCount = 0;

    output.textContent = `${minimumCoverage}%`;
    rows.forEach((row) => {
      const searchMatches = row.dataset.search.toLowerCase().includes(searchText);
      const nodeMatches = !selectedNode || row.dataset.nodes.split(';').includes(selectedNode);
      const coverageMatches = Number(row.dataset.coverage) >= minimumCoverage;
      const visible = searchMatches && nodeMatches && coverageMatches;
      row.hidden = !visible;
      if (visible) visibleCount += 1;
    });

    emptyMessage.hidden = visibleCount !== 0;
  }

  [search, node, coverage].forEach((element) => element.addEventListener('input', applyFilters));
  applyFilters();
})();
