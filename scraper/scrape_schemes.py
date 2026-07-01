import asyncio
import json
import random
import time
from playwright.async_api import async_playwright

BASE = "https://www.myscheme.gov.in"


# =========================
# ✅ Aggresive scrolling
# =========================
async def human_scroll(page):
    for _ in range(80):  # increased scroll cycles
        await page.mouse.wheel(0, random.randint(500, 1200))
        await page.wait_for_timeout(random.randint(500, 1200))

        # force bottom loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)


# =========================
# ✅ Collecting URLs
# =========================
async def get_scheme_urls(page):
    urls = set()

    print("🚀 Opening search page...")
    await page.goto(f"{BASE}/search")
    await page.wait_for_timeout(5000)

    # ✅ force initial load
    for _ in range(5):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(3000)

    prev_count = 0

    # ✅ more rounds → more data
    for i in range(10):
        print(f"\n🔄 Round {i+1}")

        await human_scroll(page)

        
        links = await page.evaluate("""
            () => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .filter(a => a.href.includes('/schemes/'))
                    .map(a => a.href);
            }
        """)

        # add unique links
        for link in links:
            urls.add(link.split("?")[0])

        print(f"📊 URLs collected: {len(urls)}")

        # ✅ click load buttons aggressively
        buttons = await page.query_selector_all("button")

        for btn in buttons:
            try:
                txt = (await btn.inner_text()).lower()

                if any(x in txt for x in ["more", "load", "view", "next"]):
                    await btn.click()
                    await page.wait_for_timeout(3000)

            except:
                pass

        # ✅ force more loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(3000)

        if len(urls) == prev_count:
            print("⚠️ No new URLs, continuing anyway...")

        prev_count = len(urls)

    return list(urls)


# =========================
# ✅ Scrapes each page
# =========================
async def scrape_scheme(context, url):
    page = await context.new_page()

    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(4000)

        name = await page.title()
        name = name.replace("| myScheme", "").strip()

        text = await page.evaluate("""
            () => {
                const main = document.querySelector('main');
                return main ? main.innerText : document.body.innerText;
            }
        """)

        print(f"✅ {name}")

        return {
            "name": name,
            "description": text[:1000],  # optimized size
            "url": url
        }

    except Exception as e:
        print(f"❌ Error: {url} → {e}")
        return None

    finally:
        await page.close()


# =========================
# ✅ Main function
# =========================
async def main():
    async with async_playwright() as p:

        browser = await p.chromium.launch(headless=False)

        context = await browser.new_context(
            viewport={"width": 1280, "height": 800}
        )

        page = await context.new_page()

        # Step 1: Collect URLs
        urls = await get_scheme_urls(page)

        print(f"\n✅ TOTAL URLs: {len(urls)}\n")

        # ✅ Step 2: scrape in batches (prevents slowdown)
        results = []

        for i in range(0, len(urls), 5):
            batch = urls[i:i+5]

            tasks = [scrape_scheme(context, url) for url in batch]
            batch_results = await asyncio.gather(*tasks)

            results.extend(batch_results)

        data = [r for r in results if r]

        await browser.close()

    # ✅ saves with timestamp
    filename = f"schemes_run_{int(time.time())}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n✅ DONE")
    print("✅ File saved:", filename)
    print("✅ Schemes scraped:", len(data))


# =========================
# ✅ Run function
# =========================
if __name__ == "__main__":
    asyncio.run(main())
