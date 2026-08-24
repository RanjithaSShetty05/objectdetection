"""
taxonomy.py — the one definition of what FieldNet's classes ARE.

WHY THIS EXISTS (2026-08-22)
============================
FieldNet's data is five source datasets pooled into one file, and their label
vocabularies are not compatible. That was invisible for months because every
diagnostic was an aggregate: instance counts, class balance, box areas and mAP
all look perfectly normal for a class whose definition is incoherent. Rendering
the boxes as contact sheets (`scripts/review_labels.py`) is what exposed it.

Two facts, both measured on the full 3385-image pool:

  1. `desk` and `table` are the SAME class under two names, and which name you
     get is decided by which dataset the photo came from. Source `indoor`
     photographs office desks with monitors, keyboards and mice on them and
     calls them `table` -- 441 boxes, never once `desk`. Source `annotation`
     photographs office desks and calls them `desk` -- 300 boxes, never once
     `table`. Source `base` uses both, and labels adjacent identical classroom
     desks `desk` and `table` IN THE SAME PHOTOGRAPH. Identical pixels,
     contradictory labels; the only cue that would disambiguate them is
     provenance, which the model cannot see. No amount of training fixes this,
     and it caps both classes' AP.

     `book` / `notebook` has the same shape: `base` labels loose sheets of paper
     and printed handouts `notebook` (479 boxes), `stationary` labels actual
     spiral notebooks `notebook` (190), and `book` collides with both.

     `pen` / `pencil` looks similar but is NOT the same problem, and is
     deliberately left alone: `base` and `stationary` BOTH use both names, and
     both distinguish them the same way. That boundary is real.

  2. Annotation completeness is a property of the SOURCE. Only `base` uses all
     16 classes. `annotation` (215 images) and `stationary` (546) use 4,
     `indoor` (498) uses 9, `object` (380) uses 11. So 1639 of 3385 images --
     48% -- come from datasets where most classes are never labelled even when
     plainly visible. PFL targets are hard 0 outside every box, so on those
     images a CORRECT detection is explicit negative supervision.

This module holds the resolution of (1): the merge map and the canonical class
list. It also holds the machinery for (2): per-source vocabularies and the
per-image class mask derived from them, so the loss can be told which channels
an image is allowed to have an opinion about.

WHY A MODULE AND NOT A CONSTANT IN EACH SCRIPT
==============================================
The class list has to agree in four places -- the split builder, the dataset,
the loss and the evaluator. It has previously disagreed, which is how
`--limit N` on train silently became invalid (the file is block-ordered by
source, so the first N images are all one source). One definition, imported.

WHAT THIS MODULE DOES NOT DO
============================
It does not drop or edit individual boxes. Per-box decisions come from a review
file produced by `scripts/review_labels.py` + `scripts/apply_label_review.py`
and are applied by `build_split.py`; they are data, not code, because they were
made by a human looking at pictures and need to stay auditable.
"""

import collections
import os
import re

# --------------------------------------------------------------------------
# Sources. The source is encoded in the filename prefix, which is the only
# provenance the pooled dataset retained.
# --------------------------------------------------------------------------
SOURCES = ("base", "stationary", "indoor", "object", "annotation")
_RF = re.compile(r"(.+?)_jpg\.rf\.[0-9a-f]+", re.I)


def source_of(file_name):
    """Which source dataset a pooled image came from. 'other' if unprefixed."""
    f = file_name.lower()
    for s in SOURCES:
        if f.startswith(s):
            return s
    return "other"


def stem_of(file_name):
    """
    The source PHOTO a file belongs to.

    Roboflow exports the same photo several times as `<stem>_jpg.rf.<hash>.jpg`
    (different crops/augments of one original). Those re-cuts must never
    straddle a split boundary or the split leaks; grouping by stem is what
    prevents it. v21 had 3 such leaks.
    """
    m = _RF.match(file_name)
    return m.group(1) if m else os.path.splitext(file_name)[0]


# --------------------------------------------------------------------------
# The classes.
# --------------------------------------------------------------------------
# The 16 names as they appear in every pooled annotations.json, in id order.
SOURCE_CLASSES = (
    "person", "chair", "desk", "table", "laptop", "mobile_phone", "book",
    "notebook", "pen", "pencil", "bottle", "bag", "keyboard", "mouse",
    "monitor", "window",
)

# Merges that resolve a name collision. Key = source name, value = the name it
# becomes. Both members of a pair must be listed, even the one whose name is
# reused, so that reading this dict tells you the whole story.
MERGES = {
    "desk": "desk_table",
    "table": "desk_table",
    "book": "book_notebook",
    "notebook": "book_notebook",
}

# Merged names are deliberately compound rather than picking a winner. If the
# merged class were just called `table`, every later reader of a per-class table
# would have to know that `table` no longer means what it means in v21/v22, and
# some of them would not. `desk_table` cannot be misread.


def canonical_classes(merge=True):
    """
    The class list to train and evaluate on, in a fixed order.

    Order is the source order with each merge group collapsed at the position
    of its first member, so the 12 untouched classes keep their familiar
    relative order and only two ids shift. Returns 14 names when merging.
    """
    if not merge:
        return list(SOURCE_CLASSES)
    out, seen = [], set()
    for name in SOURCE_CLASSES:
        dst = MERGES.get(name, name)
        if dst not in seen:
            seen.add(dst)
            out.append(dst)
    return out


def target_name(name, merge=True):
    """Source class name -> canonical class name."""
    return MERGES.get(name, name) if merge else name


def name_to_id(merge=True):
    return {n: i for i, n in enumerate(canonical_classes(merge))}


def categories(merge=True):
    """COCO-style category records for the output annotations.json."""
    return [{"id": i, "name": n, "supercategory": "none"}
            for i, n in enumerate(canonical_classes(merge))]


def remap_from(src_categories, merge=True):
    """
    Build {source category_id -> canonical category id}.

    Takes the source categories rather than trusting SOURCE_CLASSES' order,
    because a silent id shift between dataset versions would remap every box to
    the wrong class and nothing downstream would notice -- the counts would all
    still be plausible. Raises instead.
    """
    n2i = name_to_id(merge)
    out = {}
    for c in src_categories:
        name = c["name"]
        dst = target_name(name, merge)
        if dst not in n2i:
            raise ValueError(
                f"source class {name!r} maps to {dst!r}, which is not in the "
                f"canonical list {sorted(n2i)}. Update MERGES/SOURCE_CLASSES.")
        out[c["id"]] = n2i[dst]
    missing = set(n2i) - {target_name(c["name"], merge) for c in src_categories}
    if missing:
        raise ValueError(f"canonical classes with no source class: "
                         f"{sorted(missing)}")
    return out


# --------------------------------------------------------------------------
# Per-source vocabularies, and the per-image class mask they imply.
# --------------------------------------------------------------------------
# A source's vocabulary is the set of canonical classes it actually labels.
# Everything outside it is UNLABELLED, not absent, and must be masked out of the
# loss rather than treated as background.
#
# Derived from counts rather than hardcoded, because a hardcoded table silently
# rots the first time the pool changes. The derivation is not free of judgement
# though: a class with zero boxes in a source might be outside its vocabulary,
# or might just never have appeared in those particular photos. On this pool the
# distinction is not close -- the zeros are exact zeros across hundreds of
# images and the smallest nonzero count is 19 -- but `min_count` exists so the
# assumption is visible and adjustable, and `marginal()` reports any class close
# enough to the line to deserve a look.

VOCAB_MIN_COUNT = 1


def source_vocab(counts, classes, min_count=VOCAB_MIN_COUNT):
    """
    counts: {source: {canonical class name: n annotations}}
    Returns {source: sorted list of class names in that source's vocabulary}.
    """
    return {src: sorted(c for c in classes
                        if counts.get(src, {}).get(c, 0) >= min_count)
            for src in counts}


def count_by_source(anns_by_file, cat_name_of):
    """
    counts[source][class name] = number of annotations, over a whole pool.

    `cat_name_of` maps an annotation's category_id to a CANONICAL class name,
    so this must be called after remapping or with a mapping that remaps.
    """
    counts = collections.defaultdict(collections.Counter)
    for fn, alist in anns_by_file.items():
        src = source_of(fn)
        for a in alist:
            counts[src][cat_name_of(a["category_id"])] += 1
    return {k: dict(v) for k, v in counts.items()}


def marginal(counts, classes, lo=1, hi=25):
    """
    Classes whose per-source count is small but nonzero -- the only cases where
    'outside the vocabulary' vs 'just rare here' is a real judgement call.
    Print this whenever a vocabulary is derived; if it is empty, the derivation
    carried no judgement at all.
    """
    out = []
    for src in sorted(counts):
        for c in classes:
            n = counts[src].get(c, 0)
            if lo <= n < hi:
                out.append((src, c, n))
    return out


def mask_for(vocab_names, classes):
    """
    Per-image class mask as a list of 0/1 ints, one per canonical class.

    1 = this image's source labels this class, so absence of a box is real
        evidence of absence and the channel may be supervised as background.
    0 = this class is never labelled by this source. The channel gets NO
        supervision from this image: not positive, not negative.

    This is the standard federated / multi-dataset detection remedy. It is NOT
    the same thing as down-weighting background (`neg_weight`), which reduces
    the penalty on ALL background in ALL channels everywhere and so pays for
    unlabelled classes by also unlearning the labelled ones.
    """
    v = set(vocab_names)
    return [1 if c in v else 0 for c in classes]


def coverage(vocab, classes):
    """Human-readable summary: how much of the label space each source covers."""
    return {src: (len(names), len(classes)) for src, names in vocab.items()}


if __name__ == "__main__":
    # Self-test. Everything here is a property of the module, not of the data,
    # so it needs no dataset and runs in milliseconds.

    src16 = list(SOURCE_CLASSES)
    merged = canonical_classes(merge=True)

    assert len(src16) == 16, src16
    assert len(merged) == 14, merged
    assert canonical_classes(merge=False) == src16

    # The merge must collapse each pair at the position of its FIRST member,
    # so relative order is preserved. Absolute indices necessarily shift: two
    # names become one at `desk`, so everything after it moves up by one, and
    # after `book` by two. That shift is exactly why merged classes get new
    # NAMES (`desk_table`, not `table`) and why no v23 number is comparable
    # to a 16-class one.
    first_of = {"desk_table": "desk", "book_notebook": "book"}
    order_src = [first_of.get(c, c) for c in merged]
    assert order_src == sorted(order_src, key=src16.index), \
        f"merge reordered the classes: {merged}"
    assert merged.index("desk_table") == 2 and merged.index("book_notebook") == 5, \
        f"unexpected merged positions: {merged}"
    assert [c for c in merged if c not in ("desk_table", "book_notebook")] == \
        [c for c in src16 if c not in ("desk", "table", "book", "notebook")]

    # pen/pencil is deliberately NOT merged.
    assert "pen" in merged and "pencil" in merged, \
        "pen/pencil must stay separate: both base and stationary use both " \
        "names and draw the boundary the same way"

    # Every source name must land somewhere, and no two must collide except
    # via MERGES.
    n2i = name_to_id(merge=True)
    for name in src16:
        assert target_name(name, merge=True) in n2i, name
    assert target_name("laptop", merge=False) == "laptop"

    # remap_from must read the SOURCE order out of the file rather than trust
    # SOURCE_CLASSES, so a shifted id list still produces the right mapping.
    shifted = [{"id": i, "name": n}
               for i, n in enumerate(reversed(src16))]
    rm = remap_from(shifted, merge=True)
    for c in shifted:
        assert rm[c["id"]] == n2i[target_name(c["name"], merge=True)]
    print(f"remap_from survived a fully reversed id order "
          f"({len(rm)} source ids -> {len(set(rm.values()))} canonical)")

    # ...and must RAISE rather than guess on an unknown or missing class.
    for bad, why in (
        ([{"id": 0, "name": "aardvark"}], "unknown source class"),
        ([{"id": i, "name": n} for i, n in enumerate(src16[:-1])],
         "a canonical class with no source class"),
    ):
        try:
            remap_from(bad, merge=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"remap_from accepted {why}")
    print("remap_from raises on unknown and on missing classes")

    # Vocabulary derivation and the mask it implies.
    counts = {
        "base": {c: 100 for c in merged},
        "stationary": {"book_notebook": 366, "pen": 263, "pencil": 303},
        "annotation": {"chair": 515, "desk_table": 300, "laptop": 19,
                       "monitor": 167},
    }
    vocab = source_vocab(counts, merged)
    assert len(vocab["base"]) == 14
    assert vocab["stationary"] == ["book_notebook", "pen", "pencil"]
    assert len(vocab["annotation"]) == 4

    m = mask_for(vocab["stationary"], merged)
    assert sum(m) == 3 and len(m) == 14
    assert m[merged.index("pen")] == 1 and m[merged.index("person")] == 0

    # An all-labelling source must produce an all-ones mask, i.e. a no-op.
    assert mask_for(vocab["base"], merged) == [1] * 14

    # marginal() is the audit hook: it must surface annotation/laptop=19 and
    # nothing that is a clean zero or a clean large count.
    marg = marginal(counts, merged)
    assert ("annotation", "laptop", 19) in marg, marg
    assert all(n >= 1 for _, _, n in marg)
    print(f"vocabularies derived; marginal cases needing judgement: {marg}")

    # source_of / stem_of
    assert source_of("indoor_042_jpg.rf.abc123.jpg") == "indoor"
    assert source_of("mystery.jpg") == "other"
    assert stem_of("base_7_jpg.rf.deadbeef.jpg") == "base_7"
    assert stem_of("base_7_jpg.rf.cafe0001.jpg") == stem_of(
        "base_7_jpg.rf.deadbeef.jpg"), \
        "two Roboflow re-cuts of one photo must share a stem or the split leaks"
    assert stem_of("plain_name.jpg") == "plain_name"

    print(f"OK: taxonomy self-test passed. {len(merged)} canonical classes: "
          f"{', '.join(merged)}")
