"""Optional end-to-end browser check against an isolated running server.

PLAYWRIGHT_BROWSERS_PATH=/tmp/stitch-playwright .venv/bin/python tests/browser_classic.py
"""
from pathlib import Path
import tempfile
import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp,sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':1440,'height':1050})
    errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto('http://127.0.0.1:8769/classic')
    page.wait_for_selector('#engine option[value=sift]',state='attached')
    page.select_option('#engine','sift')
    assert page.locator('#weights-field').is_hidden()
    assert page.locator('[data-engine-option=ratio_test]').count()==1
    page.select_option('#preset','best')
    assert page.locator('[data-engine-option=max_keypoints]').input_value()=='2048'
    page.check('#engine-no-superpoint')
    assert page.locator('#engine option[value=superglue]').count()==0
    page.uncheck('#engine-no-superpoint')
    page.fill('#engine-search','LoMa')
    page.select_option('#engine','loma')
    assert 'checkpoint' in page.locator('#engine-status').inner_text()
    assert page.locator('#engine-verify').is_disabled()
    page.fill('#engine-search','')
    page.select_option('#engine','sift')
    scene=cv2.GaussianBlur(np.random.default_rng(19).integers(0,256,(240,650,3),dtype=np.uint8),(3,3),0)
    paths=[]
    for i,image in enumerate([scene[:,:420],scene[:,230:]]):
        path=Path(tmp)/f'tile_r00_c{i:02d}.png';cv2.imwrite(str(path),image);paths.append(str(path))
    page.set_input_files('#file-input',paths)
    page.wait_for_function('!document.getElementById("stitch-btn").disabled')
    page.select_option('#alignment','translation')
    page.select_option('#stitching-mode','grid')
    page.click('#stitch-btn')
    page.wait_for_selector('#result-card:visible',timeout=60000)
    page.click('#match-debug-open')
    page.wait_for_selector('#match-debug[open]')
    page.select_option('#match-debug-mode','inliers')
    assert '-inliers.jpg' in page.locator('#match-debug-image').get_attribute('src')
    assert page.locator('#match-debug-artifact').is_visible()
    page.click('#pair-retry-current')
    page.wait_for_selector('#pair-retry-results article',timeout=60000)
    assert 'accepted' in page.locator('#pair-retry-results').inner_text()
    (ROOT/'validation/classic_ml').mkdir(parents=True,exist_ok=True)
    page.screenshot(path=str(ROOT/'validation/classic_ml/pair-inspector.png'))
    page.click('#match-debug-close')
    page.fill('#project-name','Browser model test')
    page.click('#save-project')
    page.wait_for_function('document.getElementById("log").innerText.includes("saved") || document.getElementById("log").innerText.includes("Saved")')
    page.evaluate('async () => { const projects = await (await fetch("/api/projects")).json(); await openProject(projects.projects.find(p => p.name === "Browser model test").id); }')
    assert page.locator('#grid-overlay-content .thumb').count()==2
    assert page.locator('#alignment').input_value()=='translation'
    assert page.locator('#engine').input_value()=='sift'
    page.goto('http://127.0.0.1:8769/compare')
    page.wait_for_selector('#model-options-sift [data-engine-option=ratio_test]')
    assert page.locator('[name=engine][value=loma]').is_disabled()
    assert page.locator('#model-options-efficientloftr [data-engine-option=dense_sample_budget]').count()==1
    page.goto('http://127.0.0.1:8769/classic')
    page.wait_for_selector('#engine option[value=sift]',state='attached')
    invalid=Path(tmp)/'unnamed.png';cv2.imwrite(str(invalid),scene)
    page.set_input_files('#file-input',[str(invalid),paths[0]])
    page.wait_for_selector('#filename-warning[open]')
    assert not errors,errors
    browser.close()
    print('Browser checks passed: catalog, filters, controls, grid run, inspector, retry, save, comparison, filename warning.')
