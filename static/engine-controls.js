/* Shared schema renderer: model choices and defaults come only from /api/engines. */
window.EngineControls = {
  async catalog() {
    const response = await fetch('/api/engines');
    if (!response.ok) throw Error('Could not load model catalog');
    return (await response.json()).engines;
  },
  render(container, entry, values = {}) {
    container.replaceChildren();
    for (const [name, spec] of Object.entries(entry.options_schema)) {
      const label = document.createElement('label');
      label.className = 'engine-option';
      const text = document.createElement('span');
      text.textContent = name.replaceAll('_', ' ');
      label.append(text);
      const input = document.createElement(spec.type === 'enum' ? 'select' : 'input');
      input.dataset.engineOption = name;
      if (spec.type === 'enum') {
        for (const value of spec.values) input.add(new Option(value, value));
      } else {
        input.type = 'number'; input.min = spec.min; input.max = spec.max;
        input.step = spec.type === 'integer' ? '1' : '0.01';
      }
      input.value = values[name] ?? spec.default;
      input.title = spec.help;
      label.append(input);
      container.append(label);
    }
  },
  read(container) {
    const values = {version: 1};
    container.querySelectorAll('[data-engine-option]').forEach(input => {
      values[input.dataset.engineOption] = input.type === 'number' ? Number(input.value) : input.value;
    });
    return values;
  },
  async verify(entry, options, status) {
    status.textContent = 'Loading model and testing translated and unrelated pairs… Downloads may take several minutes.';
    const response = await fetch(`/api/engines/${entry.id}/verify`, {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(options),
    });
    const result = await response.json();
    if (!response.ok || !result.passed) throw Error(result.detail || result.reason || 'Verification failed');
    status.textContent = `Verified on ${result.device}: ${result.inliers} inliers, ${result.translation_error_px.toFixed(2)} px error. This is a smoke test, not a guarantee for your images.`;
    return result;
  },
};
