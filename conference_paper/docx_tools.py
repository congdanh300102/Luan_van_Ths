"""
Helpers for editing the SEBL2026 manuscript in place.

The manuscript is a WPS-authored .docx whose paragraph and table styling is
carried on individual runs and cell properties rather than on named styles, so
every edit here works by REUSING the existing run/cell XML and swapping only
the text. Creating fresh paragraphs or tables would silently drop the journal
template's fonts, borders and spacing.
"""
from copy import deepcopy

from docx.oxml.ns import qn
from docx.shared import Pt


# ---------------------------------------------------------------------------
# Paragraph-level editing
# ---------------------------------------------------------------------------
def strip_hyperlinks(p):
    """Unwrap every w:hyperlink in the paragraph, keeping its visible text.

    Used for the Data Availability statement, which embedded a live Google
    Drive link to the confidential source file. Deleting only the visible text
    would leave the relationship in word/_rels intact and the URL recoverable,
    so the w:hyperlink element itself has to go.
    """
    removed = []
    for hl in p._p.findall(qn('w:hyperlink')):
        rid = hl.get(qn('r:id'))
        if rid:
            removed.append(rid)
        p._p.remove(hl)
    return removed


def _base_rpr(p):
    """rPr of the first real run, used as the formatting template."""
    for r in p._p.findall(qn('w:r')):
        rpr = r.find(qn('w:rPr'))
        if rpr is not None:
            return deepcopy(rpr)
    return None


def set_rich(p, segments):
    """Replace a paragraph's content with `segments` = [(text, bold), ...].

    Formatting is inherited from the paragraph's current first run, so the
    replacement keeps the manuscript's body font and size.
    """
    rpr = _base_rpr(p)
    strip_hyperlinks(p)
    for r in p._p.findall(qn('w:r')):
        p._p.remove(r)

    for text, bold in segments:
        if text == "":
            continue
        run = p._p.makeelement(qn('w:r'), {})
        if rpr is not None:
            new_rpr = deepcopy(rpr)
            for tag in ('w:b', 'w:bCs'):
                for el in new_rpr.findall(qn(tag)):
                    new_rpr.remove(el)
            if bold:
                for tag in ('w:b', 'w:bCs'):
                    new_rpr.append(new_rpr.makeelement(qn(tag), {}))
            run.append(new_rpr)
        t = run.makeelement(qn('w:t'), {})
        t.set(qn('xml:space'), 'preserve')
        t.text = text
        run.append(t)
        p._p.append(run)
    return p


def set_text(p, text):
    """Replace a paragraph's text, optionally with **bold** markup."""
    segments = []
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            segments.append((chunk, i % 2 == 1))
    return set_rich(p, segments or [("", False)])


def insert_paragraph_after(p, text):
    """Clone `p` (keeping its formatting) and insert the copy right after it."""
    new_p = deepcopy(p._p)
    p._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    para = Paragraph(new_p, p._parent)
    set_text(para, text)
    return para


# ---------------------------------------------------------------------------
# Table-level editing
# ---------------------------------------------------------------------------
def _set_grid(tbl, n_cols, total_width=9072):
    """Resize w:tblGrid to n_cols equal columns."""
    grid = tbl._tbl.find(qn('w:tblGrid'))
    cols = grid.findall(qn('w:gridCol'))
    while len(cols) < n_cols:
        grid.append(deepcopy(cols[-1]))
        cols = grid.findall(qn('w:gridCol'))
    while len(cols) > n_cols:
        grid.remove(cols[-1])
        cols = grid.findall(qn('w:gridCol'))
    width = total_width // n_cols
    for c in cols:
        c.set(qn('w:w'), str(width))
    return width


def _fit_row(tr, n_cols, cell_width):
    """Make a w:tr have exactly n_cols cells, cloning the last one as needed."""
    tcs = tr.findall(qn('w:tc'))
    while len(tcs) < n_cols:
        tr.append(deepcopy(tcs[-1]))
        tcs = tr.findall(qn('w:tc'))
    while len(tcs) > n_cols:
        tr.remove(tcs[-1])
        tcs = tr.findall(qn('w:tc'))
    for tc in tcs:
        tcpr = tc.find(qn('w:tcPr'))
        if tcpr is not None:
            for gs in tcpr.findall(qn('w:gridSpan')):
                tcpr.remove(gs)
            tcw = tcpr.find(qn('w:tcW'))
            if tcw is None:
                tcw = tcpr.makeelement(qn('w:tcW'), {})
                tcpr.insert(0, tcw)
            tcw.set(qn('w:w'), str(cell_width))
            tcw.set(qn('w:type'), 'dxa')
    return tcs


def set_table(tbl, rows, n_header=1, total_width=9072):
    """Rewrite a table's contents, reusing its existing row/cell formatting.

    `rows` is a list of lists of strings; row 0..n_header-1 are treated as
    header rows and keep the header row's formatting (bold, shading, repeat-on-
    page-break), all later rows reuse the first data row's formatting.
    """
    n_cols = max(len(r) for r in rows)
    cell_width = _set_grid(tbl, n_cols, total_width)

    trs = tbl._tbl.findall(qn('w:tr'))
    header_tpl = deepcopy(trs[0])
    data_tpl = deepcopy(trs[1] if len(trs) > 1 else trs[0])
    for tr in trs:
        tbl._tbl.remove(tr)

    for i, row in enumerate(rows):
        tr = deepcopy(header_tpl if i < n_header else data_tpl)
        tcs = _fit_row(tr, n_cols, cell_width)
        for j, tc in enumerate(tcs):
            value = row[j] if j < len(row) else ""
            _set_cell_text(tc, value)
        tbl._tbl.append(tr)
    return tbl


def _set_cell_text(tc, text):
    """Set a w:tc's text, keeping the first paragraph's run formatting."""
    ps = tc.findall(qn('w:p'))
    for p in ps[1:]:
        tc.remove(p)
    p = ps[0]

    rpr = None
    for r in p.findall(qn('w:r')):
        found = r.find(qn('w:rPr'))
        if found is not None:
            rpr = deepcopy(found)
            break
    for r in p.findall(qn('w:r')):
        p.remove(r)
    for hl in p.findall(qn('w:hyperlink')):
        p.remove(hl)

    bold = text.startswith("**") and text.endswith("**") and len(text) > 4
    if bold:
        text = text[2:-2]

    run = p.makeelement(qn('w:r'), {})
    if rpr is not None:
        new_rpr = deepcopy(rpr)
        for tag in ('w:b', 'w:bCs'):
            for el in new_rpr.findall(qn(tag)):
                new_rpr.remove(el)
        if bold:
            for tag in ('w:b', 'w:bCs'):
                new_rpr.append(new_rpr.makeelement(qn(tag), {}))
        run.append(new_rpr)
    t = run.makeelement(qn('w:t'), {})
    t.set(qn('xml:space'), 'preserve')
    t.text = text
    run.append(t)
    p.append(run)


# ---------------------------------------------------------------------------
# Image replacement
# ---------------------------------------------------------------------------
def image_parts_in_order(doc):
    """Return image parts in the order their <w:drawing> appears in the body."""
    seen, ordered = set(), []
    for blip in doc.element.body.iter(qn('a:blip')):
        rid = blip.get(qn('r:embed'))
        if rid and rid not in seen:
            seen.add(rid)
            ordered.append((rid, doc.part.related_parts[rid]))
    return ordered


def replace_image(doc, index, png_path):
    """Swap the bytes of the index-th inline image (0-based, document order)."""
    rid, part = image_parts_in_order(doc)[index]
    with open(png_path, 'rb') as fh:
        part._blob = fh.read()
    return rid
