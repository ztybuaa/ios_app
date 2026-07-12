#!/usr/bin/env python3
"""Regression checks for converting extracted resource spans into search plans."""

from __future__ import annotations

from prepare_dataset import extract_qualifiers, normalize_search_keyword


CASES = [
    ("photo", "小猫图片", "小猫", [], [], []),
    ("photo", "Tripadvisor照片", "Tripadvisor", [], [], []),
    ("photo", "截图", "截图", [], [], []),
    ("photo", "拍立得照片", "拍立得", [], [], []),
    ("photo", "今天拍的照片", None, ["今天"], [], []),
    ("video", "最新的视频", None, ["最新"], [], ["recent"]),
    ("video", "录屏视频", "录屏", [], [], []),
    ("video", "刚才录的会议视频", "会议", ["刚才"], [], ["recent"]),
    ("video", "MP4格式的视频", None, [], ["MP4"], []),
    ("video", "mp4视频", None, [], ["MP4"], []),
    ("file", "最近下载的文件", None, ["最近"], [], ["recent"]),
    ("file", "PDF报告", "报告", [], ["PDF"], []),
    ("file", "合同文档", "合同", [], [], []),
    ("folder", "上周的文件夹", None, ["上周"], [], []),
    ("folder", "合同归档目录", "合同归档", [], [], []),
    ("contact", "今天新增的联系人", None, ["今天"], [], ["recent"]),
    ("contact", "小明的手机号", "小明", [], [], []),
]


def main() -> None:
    for intent, phrase, keyword, times, formats, selection_hints in CASES:
        actual_keyword = normalize_search_keyword(intent, phrase)
        qualifiers = extract_qualifiers(phrase)
        assert actual_keyword == keyword, (
            f"{intent}/{phrase}: expected keyword {keyword!r}, got {actual_keyword!r}"
        )
        assert qualifiers["time"] == times, (
            f"{intent}/{phrase}: expected time {times!r}, got {qualifiers['time']!r}"
        )
        assert qualifiers["format"] == formats, (
            f"{intent}/{phrase}: expected format {formats!r}, got {qualifiers['format']!r}"
        )
        assert qualifiers["selection_hint"] == selection_hints, (
            f"{intent}/{phrase}: expected selection hints {selection_hints!r}, "
            f"got {qualifiers['selection_hint']!r}"
        )

    print(f"Resource query normalization validation passed ({len(CASES)} cases)")


if __name__ == "__main__":
    main()
