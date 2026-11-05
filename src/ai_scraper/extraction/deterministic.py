"""Conservative JSON-LD and WooCommerce selectors; no model-generated facts."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from ..schemas import ISO_CURRENCIES, EvidenceCandidate, ExtractionCandidate, ProductV1, SourceSnapshot


def parse_price(text: str, locale: str | None = None) -> str | None:
    """Normalize a single amount; grouping/decimal commas require an explicit locale."""
    numbers = re.findall(r"(?<![\w])\d[\d.,\u00a0 ]*(?:\d)?", text)
    numbers = [n.strip() for n in numbers if n.strip()]
    if len(numbers) != 1 or re.search(r"[-−]\s*\d", text):
        return None
    value = numbers[0].replace("\u00a0", " ")
    if locale in {"de_DE", "fr_FR"}:
        value = value.replace("." if locale == "de_DE" else " ", "").replace(",", ".")
    elif locale in {"en_US", "en_GB"}:
        if "," in value and not re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", value):
            return None
        value = value.replace(",", "")
    elif "," in value or " " in value:
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", value):
        return None
    try:
        amount = Decimal(value)
    except InvalidOperation:
        return None
    return value if amount.is_finite() and amount >= 0 else None


def currency_from_text(text: str) -> str | None:
    codes = set(re.findall(r"(?<![A-Z])[A-Z]{3}(?![A-Z])", text)) & ISO_CURRENCIES
    if len(codes) == 1:
        return codes.pop()
    if len(codes) > 1:
        return None
    # These symbols have a unique ISO interpretation. '$' and '¥' do not.
    if "€" in text and "£" not in text:
        return "EUR"
    if "£" in text and "€" not in text:
        return "GBP"
    return None


def _canonical_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, "")
    )


def _walk(value: Any, path: str):
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from _walk(child, f"{path}/{escaped}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _walk(child, f"{path}/{i}")


def _scalar_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_UNRELATED_HTML_CLASSES = frozenset(
    {
        "related",
        "related-products",
        "upsells",
        "up-sells",
        "cross-sells",
        "recommendations",
        "recommended-products",
        "recently-viewed",
        "cart-contents",
        "site-header-cart",
        "widget_shopping_cart",
    }
)


def _unrelated_html(node) -> bool:
    return any(
        _UNRELATED_HTML_CLASSES.intersection(parent.get("class", []))
        for parent in (node, *node.parents)
        if hasattr(parent, "get")
    )


def _primary_price_scope(soup, headings):
    """Tie visible offers to the single primary product heading, not page chrome.

    WooCommerce places the primary heading and offer in .summary/.entry-summary.
    A product container is a fallback for themes that separate the heading from
    its summary. Without that identity anchor, retain conservative global
    matching after filtering recommendations and cart widgets.
    """
    if len(headings) != 1:
        return soup
    heading = headings[0]
    for ancestor in heading.parents:
        if {"summary", "entry-summary"}.intersection(ancestor.get("class", [])):
            return ancestor
    for ancestor in heading.parents:
        if "product" in ancestor.get("class", []):
            return ancestor
    return soup


def deterministic_extract(snapshot: SourceSnapshot, locale: str | None = None) -> ExtractionCandidate:
    values: dict[str, Any] = ProductV1().model_dump()
    evidence: list[EvidenceCandidate] = []
    warnings: list[str] = []
    products = []
    for i, root in enumerate(snapshot.ld_json):
        for pointer, obj in _walk(root, f"/{i}"):
            types = obj.get("@type", [])
            if isinstance(types, str):
                types = [types]
            if "Product" in types:
                products.append((pointer, obj))
    target = {_canonical_url(snapshot.requested_url), _canonical_url(snapshot.final_url)}

    def matches(obj: dict) -> bool:
        identity = obj.get("url") or obj.get("@id")
        return isinstance(identity, str) and _canonical_url(urljoin(snapshot.final_url, identity)) in target

    selected = [x for x in products if matches(x[1])]
    if len(selected) == 1:
        product = selected[0]
    elif len(products) == 1 and not (products[0][1].get("url") or products[0][1].get("@id")):
        product = products[0]
    else:
        product = None
        if products:
            warnings.append("ambiguous_entity: no unique Product matches the requested URL")

    def set_ld(field: str, value: Any, pointer: str) -> None:
        if value is None or value == "":
            return
        parts = field.split(".")
        if len(parts) == 1:
            values[field] = value
        elif parts[0] == "images":
            values["images"].append(value)
        elif parts[0] == "category":
            values["category"] = values["category"] or {"name": None, "url": None}
            values["category"][parts[1]] = value
        evidence.append(
            EvidenceCandidate(field_path=field, json_pointer=pointer, excerpt=_scalar_text(value))
        )

    if product:
        pointer, obj = product
        for field, key in (("name", "name"), ("sku", "sku"), ("long_description", "description")):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                set_ld(field, value, f"{pointer}/{key}")
        category = obj.get("category")
        if isinstance(category, str) and category:
            set_ld("category.name", category, f"{pointer}/category")
        elif isinstance(category, dict):
            for field in ("name", "url"):
                if isinstance(category.get(field), str):
                    set_ld(f"category.{field}", category[field], f"{pointer}/category/{field}")
        image = obj.get("image")
        images = image if isinstance(image, list) else [image]
        for i, item in enumerate(images):
            ip = f"{pointer}/image" + (f"/{i}" if isinstance(image, list) else "")
            if isinstance(item, dict):
                key = "url" if item.get("url") else "contentUrl"
                item, ip = item.get(key), f"{ip}/{key}"
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                set_ld(f"images.{len(values['images'])}", item, ip)
        offers = obj.get("offers")
        offer_items = offers if isinstance(offers, list) else [offers]
        offer_items = [x for x in offer_items if isinstance(x, dict)]
        prices = {str(x["price"]) for x in offer_items if x.get("price") is not None}
        currencies = {x["priceCurrency"] for x in offer_items if isinstance(x.get("priceCurrency"), str)}
        if len(prices) > 1:
            warnings.append("conflict:price: multiple offer prices")
        elif len(prices) == 1:
            amount = prices.pop()
            if parse_price(amount) is not None:
                index = next(i for i, x in enumerate(offer_items) if str(x.get("price")) == amount)
                op = f"{pointer}/offers" + (f"/{index}" if isinstance(offers, list) else "")
                set_ld("price", amount, f"{op}/price")
                set_ld("price_text", amount, f"{op}/price")
        if len(currencies) > 1:
            warnings.append("conflict:currency: multiple offer currencies")
        elif len(currencies) == 1:
            currency = currencies.pop()
            if currency in ISO_CURRENCIES:
                index = next(i for i, x in enumerate(offer_items) if x.get("priceCurrency") == currency)
                op = f"{pointer}/offers" + (f"/{index}" if isinstance(offers, list) else "")
                set_ld("currency", currency, f"{op}/priceCurrency")

    soup = BeautifulSoup(snapshot.raw_record.get("page_html") or "", "html.parser")
    for item in soup.select("script, style, noscript, template"):
        item.decompose()

    def text_evidence(field: str, value: str, source: str) -> None:
        for section in snapshot.sections:
            if source in section.text:
                evidence.append(EvidenceCandidate(field_path=field, section_id=section.id, excerpt=source))
                return

    def set_text(field: str, value: str | None, source: str) -> None:
        if value is None:
            return
        existing = values.get(field)
        if existing is not None and existing != value:
            equal_price = field == "price" and Decimal(existing) == Decimal(value)
            if not equal_price:
                warnings.append(f"conflict:{field}: structured data differs from visible content")
                values[field] = None
                return
        values[field] = value
        evidence[:] = [e for e in evidence if e.field_path != field]
        text_evidence(field, value, source)

    if not products or product:
        names = [
            node
            for node in soup.select("h1.product_title, h1[itemprop=name], h1.product-title")
            if not _unrelated_html(node)
        ]
        if not names:
            all_names = [node for node in soup.select("h1") if not _unrelated_html(node)]
            names = all_names if len(all_names) == 1 else []
        if len(names) == 1:
            name = names[0].get_text(" ", strip=True)
            set_text("name", name or None, name)
        price_scope = _primary_price_scope(soup, names)
        price_nodes = [
            node
            for node in price_scope.select(".price, .product-price, [itemprop=price]")
            if not _unrelated_html(node)
        ]
        # Nested matching nodes can be the same price; distinct prices are ambiguous.
        texts = list(dict.fromkeys(n.get("content") or n.get_text(" ", strip=True) for n in price_nodes))
        texts = [x for x in texts if x]
        amounts = {parse_price(x, locale) for x in texts} - {None}
        multiple_amounts_in_one_node = any(
            len(
                {
                    parse_price(part.strip(), locale)
                    for part in re.findall(r"(?<![\w])\d[\d.,\u00a0 ]*(?:\d)?", text)
                }
                - {None}
            )
            > 1
            for text in texts
        )
        if (
            len(amounts) > 1
            or multiple_amounts_in_one_node
            or any(re.search(r"\d.*(?:–| - ).*\d", x) for x in texts)
        ):
            warnings.append("conflict:price: multiple visible prices or variants")
            values["price"] = None
            if texts:
                values["price_text"] = texts[0]
                evidence[:] = [e for e in evidence if e.field_path != "price_text"]
                text_evidence("price_text", texts[0], texts[0])
                set_text("currency", currency_from_text(texts[0]), texts[0])
        elif texts:
            price_text = texts[0]
            # Price text is the visible verbatim source, intentionally distinct from JSON-LD amount.
            values["price_text"] = price_text
            evidence[:] = [e for e in evidence if e.field_path != "price_text"]
            text_evidence("price_text", price_text, price_text)
            amount = parse_price(price_text, locale)
            set_text("price", amount, price_text)
            currency = currency_from_text(price_text)
            if currency:
                set_text("currency", currency, price_text)
        for field, selector in (
            ("sku", ".sku"),
            ("short_description", ".woocommerce-product-details__short-description"),
            ("long_description", "#tab-description"),
        ):
            node = soup.select_one(selector)
            if node:
                value = node.get_text(" ", strip=True)
                # Blocks can span saved lines; require the retained block representation.
                if value and any(value in s.text for s in snapshot.sections):
                    set_text(field, value, value)
        if not values["category"]:
            category_node = soup.select_one(".posted_in a")
            if category_node:
                name = category_node.get_text(" ", strip=True)
                link = category_node.get("href")
                if name:
                    values["category"] = {"name": name, "url": None}
                    text_evidence("category.name", name, name)
                    if isinstance(link, str) and link.startswith(("http://", "https://")):
                        values["category"]["url"] = link
                        text_evidence("category.url", link, link)
        if not values["images"]:
            for image_node in soup.select(".woocommerce-product-gallery__image img"):
                link = image_node.get("data-src") or image_node.get("src")
                if (
                    isinstance(link, str)
                    and link.startswith(("http://", "https://"))
                    and link not in values["images"]
                ):
                    field = f"images.{len(values['images'])}"
                    values["images"].append(link)
                    text_evidence(field, link, link)
        for attribute in ("size", "color"):
            node = soup.select_one(f".woocommerce-product-attributes-item--attribute_pa_{attribute} td")
            if node:
                text = node.get_text(" ", strip=True)
                parts = [part.strip() for part in text.split(",") if part.strip()]
                if parts:
                    values["additional_info"] = values["additional_info"] or {"size": [], "color": []}
                    values["additional_info"][attribute] = parts
                    for index, part in enumerate(parts):
                        text_evidence(f"additional_info.{attribute}.{index}", part, text)
        currency_node = soup.select_one(
            '[itemprop=priceCurrency][content], meta[property="product:price:currency"]'
        )
        if currency_node and currency_node.get("content") in ISO_CURRENCIES:
            code = currency_node["content"]
            set_text("currency", code, code)

    conflict_fields = {w.split(":", 2)[1] for w in warnings if w.startswith("conflict:")}
    for field in conflict_fields:
        if field in values:
            values[field] = None
    evidence = [e for e in evidence if e.field_path not in conflict_fields]
    if not snapshot.ld_json:
        warnings.append("representation: JSON-LD unavailable")
    return ExtractionCandidate(data=ProductV1.model_validate(values), evidence=evidence, warnings=warnings)
