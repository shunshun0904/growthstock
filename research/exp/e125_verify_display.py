#!/usr/bin/env python3
"""Verify the saved E125 report with JavaScript disabled and network offline."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    ap.add_argument('--chromium',default='/usr/bin/chromium')
    a=ap.parse_args();p=a.output
    html=(p/'growthstock_tsfm_exit.html').read_text()
    results=[]
    with sync_playwright() as pw:
        browser=pw.chromium.launch(executable_path=a.chromium,headless=True,args=['--no-sandbox'])
        for width,height in [(1440,1000),(390,844)]:
            context=browser.new_context(viewport={'width':width,'height':height},
                                       java_script_enabled=False,offline=True,service_workers='block')
            page=context.new_page();external=[]
            page.on('request',lambda r:external.append(r.url) if r.url.startswith(('http:','https:')) else None)
            page.set_content(html,wait_until='load')
            assert page.locator('script').count()==0
            assert page.locator('table').count()==71
            assert page.locator('details').count()==60
            assert page.locator('img').count()==3
            for text in ['購入後の上値予測','0.1503','0.1193','9752','追加学習なし','除外5件']:
                assert page.get_by_text(text,exact=False).count()>0,text
            assert page.locator('h1').is_visible()
            # The probability summary and every aggregate table are visible.
            all_tables=page.locator('table')
            shown=0
            for i in range(all_tables.count()):
                t=all_tables.nth(i)
                if not t.evaluate('(el)=>!!el.closest("details")'):
                    assert t.is_visible();shown+=1
            assert shown==11
            for image in page.locator('img').all():
                assert image.evaluate('(el)=>el.complete && el.naturalWidth>0')
            page.screenshot(path=str(p/f'preview-{width}.png'),full_page=False)
            first=page.locator('details').first
            first.locator('summary').click()
            assert first.get_attribute('open') is not None
            assert first.locator('tbody tr').count()==19
            assert first.locator('table').is_visible()
            assert first.get_by_text('最高値中央値%',exact=True).is_visible()
            page.screenshot(path=str(p/f'preview-trade-{width}.png'),full_page=False)
            body_overflow=page.evaluate('document.documentElement.scrollWidth > window.innerWidth+2')
            assert not body_overflow,'Tables must scroll inside their container on mobile'
            assert not external,'Offline artifact must not request remote assets'
            results.append(dict(width=width,aggregate_tables_visible=shown,trade_details=60,
                                opened_trade_rows=19,embedded_images=3,body_overflow=body_overflow))
            context.close()
        browser.close()
    result=dict(all_passed=True,javascript_enabled=False,network_offline=True,viewports=results)
    (p/'display_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
