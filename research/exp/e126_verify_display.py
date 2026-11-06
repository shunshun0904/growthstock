#!/usr/bin/env python3
"""Check all E126 aggregates and native disclosure tables offline, without JS."""
import json
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

P=Path(sys.argv[1])
body=(P/'growthstock_optimal_exit.html').read_text()
results=[]
with sync_playwright() as pw:
    browser=pw.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
    for width,height in [(1440,1000),(390,844)]:
        context=browser.new_context(viewport={'width':width,'height':height},java_script_enabled=False,
                                   offline=True,service_workers='block')
        page=context.new_page();external=[]
        page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
        page.set_content(body,wait_until='load')
        assert page.locator('script').count()==0
        assert page.locator('table').count()==88
        assert page.locator('details').count()==69
        assert page.locator('details tbody tr').count()==69*19
        assert page.locator('img').count()==2
        for text in ['購入後の出口戦略','3,265','0.1318','20日保有','購入後も固定','最適停止']:
            assert page.get_by_text(text,exact=False).count()>0,text
        assert page.locator('h1').is_visible()
        shown=0
        for table in page.locator('table').all():
            if not table.evaluate('(el)=>!!el.closest("details")'):
                assert table.is_visible();shown+=1
        assert shown==19
        for image in page.locator('img').all():
            assert image.evaluate('(el)=>el.complete && el.naturalWidth>0')
        page.screenshot(path=str(P/f'preview-{width}.png'),full_page=False)
        first=page.locator('details').first;first.locator('summary').click()
        assert first.get_attribute('open') is not None and first.locator('table').is_visible()
        assert first.get_by_text('最高値中央値%',exact=True).is_visible()
        page.screenshot(path=str(P/f'preview-trade-{width}.png'),full_page=False)
        overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth+2')
        assert not overflow and not external
        results.append(dict(width=width,aggregate_tables_visible=shown,trade_details=69,
                            daily_rows=69*19,embedded_images=2,body_overflow=overflow))
        context.close()
    browser.close()
result=dict(all_passed=True,javascript_enabled=False,network_offline=True,viewports=results)
(P/'display_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
