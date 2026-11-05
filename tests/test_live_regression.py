"""Small reviewed WooCommerce DOM reproductions from the article-page smoke test."""

import json

import pytest

from ai_scraper.extraction.deterministic import deterministic_extract
from ai_scraper.normalize import normalize_record
from ai_scraper.schemas import ExtractionRequest
from ai_scraper.validate import validate_candidate

URL = "https://www.scrapingcourse.com/ecommerce/product/abominable-hoodie/"


def extract(visible_price, *, structured_price="69.00", summary_class="summary entry-summary"):
    product = {
        "@type": "Product",
        "name": "Abominable Hoodie",
        "url": URL,
        "offers": {"price": structured_price, "priceCurrency": "USD"},
    }
    html = (
        '<header><a class="cart-contents"><span itemprop="price">$ 0.00</span></a></header>'
        '<div class="product"><div class="' + summary_class + '">'
        '<h1 class="product_title">Abominable Hoodie</h1>' + visible_price + "</div>"
        '<section class="related products"><ul><li class="product">'
        '<h2>Different Hoodie</h2><span class="price"><span itemprop="price">$ 42.00</span></span>'
        '</li><li class="product"><span class="price" itemprop="price">$ 62.00</span></li>'
        '</ul></section></div><script type="application/ld+json">' + json.dumps(product) + "</script>"
    )
    snapshot = normalize_record({"input": {"url": URL}, "page_html": html}, ExtractionRequest(url=URL))
    candidate = deterministic_extract(snapshot)
    return candidate, validate_candidate(candidate, snapshot)


@pytest.mark.parametrize("summary_class", ["summary entry-summary", "entry-summary", "summary"])
def test_primary_price_ignores_cart_and_related_product_offers(summary_class):
    candidate, assessment = extract(
        '<p class="price"><span itemprop="price">$ 69.00</span></p>', summary_class=summary_class
    )
    assert assessment.status == "success", assessment.errors
    assert candidate.data.name == "Abominable Hoodie"
    assert candidate.data.price == "69.00"
    assert candidate.data.price_text == "$ 69.00"
    assert candidate.data.currency == "USD"
    assert not any(warning.startswith("conflict:price:") for warning in candidate.warnings)
    assert assessment.evidence["price"].excerpt == "$ 69.00"


def test_multiple_primary_variant_prices_still_conflict():
    candidate, assessment = extract('<p class="price">$ 69.00</p><p class="price">$ 79.00</p>')
    assert assessment.status == "partial"
    assert candidate.data.price is None
    assert any(warning.startswith("conflict:price:") for warning in candidate.warnings)


def test_sale_and_original_prices_in_one_primary_node_still_conflict():
    candidate, assessment = extract('<p class="price"><del>$ 79.00</del><ins>$ 69.00</ins></p>')
    assert assessment.status == "partial"
    assert candidate.data.price is None
    assert any(warning.startswith("conflict:price:") for warning in candidate.warnings)


def test_primary_visible_and_jsonld_mismatch_still_conflicts():
    candidate, assessment = extract('<p class="price">$ 59.00</p>')
    assert assessment.status == "partial"
    assert candidate.data.price is None
    assert any("structured data differs" in warning for warning in candidate.warnings)
