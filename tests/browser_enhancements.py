"""Run against an isolated server on port 8769; uses installed Chrome."""
from pathlib import Path
import tempfile

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
    page = browser.new_page(viewport={'width':1440,'height':1050})
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto('http://127.0.0.1:8769/classic')
    page.wait_for_selector('#engine option[value=sift]', state='attached')
    for key in ('flatfield','global-adjustment','guided-retry','grid-priors'):
        assert not page.locator('#'+key).is_checked()
    assert page.locator('#global-adjustment').is_disabled()
    page.select_option('#stitching-mode','grid')
    for key in ('global-adjustment','guided-retry','grid-priors'):
        page.check('#'+key)
    page.select_option('#alignment','homography')
    assert not page.locator('#global-adjustment').is_checked()
    assert page.locator('#guided-retry').is_disabled()
    page.select_option('#alignment','translation')
    page.check('#flatfield')
    page.select_option('#preprocessing','sobel')
    assert page.locator('#flatfield').is_disabled()
    assert not page.locator('#flatfield').is_checked()
    page.select_option('#preprocessing','none')
    for key in ('flatfield','global-adjustment','guided-retry','grid-priors'):
        page.check('#'+key)
    page.select_option('#engine','sift')
    scene = cv2.GaussianBlur(np.random.default_rng(19).integers(0,256,(240,650,3),dtype=np.uint8),(3,3),0)
    paths = []
    for i, image in enumerate([scene[:,:420], scene[:,230:]]):
        path = Path(tmp)/f'tile_r00_c{i:02d}.png'
        cv2.imwrite(str(path), image)
        paths.append(str(path))
    page.set_input_files('#file-input',paths)
    page.wait_for_function('!document.getElementById("stitch-btn").disabled')
    page.click('#stitch-btn')
    page.wait_for_selector('#result-card:visible',timeout=60000)
    page.fill('#project-name','Optional correction browser test')
    with page.expect_response(lambda response: response.url.endswith('/api/projects') and response.request.method == 'POST') as saved:
        page.click('#save-project')
    project_id = saved.value.json()['id']
    try:
        for key in ('flatfield','global-adjustment','guided-retry','grid-priors'):
            page.uncheck('#'+key)
        page.evaluate('(id) => openProject(id)', project_id)
        for key in ('flatfield','global-adjustment','guided-retry','grid-priors'):
            assert page.locator('#'+key).is_checked()
        # Check the visible disclosure without requiring a deliberately broken matcher.
        page.evaluate('showResult({image:state.resultUrl,meta:{graph:{predicted_edges:[{}]}}})')
        assert 'not verified image matches' in page.locator('#result-notice').inner_text()
        assert not errors, errors
        print('Browser checks passed: defaults, mode gating, batch job, saved options, predicted-link notice.')
    finally:
        page.request.delete('http://127.0.0.1:8769/api/projects/'+project_id)
        browser.close()
