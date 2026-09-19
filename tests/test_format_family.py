# ABOUTME: ParsedRecord fields + FormatFamily grouping/registry

from jaff.core.parsers.network._formats import (
    ParsedRecord,
    FormatFamily,
    all_families,
)


def test_parsed_record_fields():
    pr = ParsedRecord(
        r=["H", "H"], p=["H2"], tmin=10.0, tmax=1000.0,
        rate="1e-10", type="unknown", string="H + H -> H2", source_index=4,
    )
    assert pr.r == ["H", "H"] and pr.source_index == 4 and pr.sub_order == 0


def test_families_group_members_by_family_attr():
    fams = {f.name: f for f in all_families()}
    krome_members = {type(m).__name__ for m in fams["krome"].members}
    assert "KromeReaction" in krome_members
    assert "KromeFormatHeader" in krome_members
    assert "kida" in fams and len(fams["kida"].members) == 1
