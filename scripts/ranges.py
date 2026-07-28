"""거리대 구간 정의와 한글 글자 그리기.

거리대 구간은 여기 한 곳에만 둔다. 리포트와 확인 영상이 서로 다른 기준을 쓰면
영상을 보며 검증하는 의미가 없어진다.

OpenCV 기본 글꼴은 한글이 안 나온다 (물음표로 깨진다). 그래서 한글은 PIL 로 그린다.
프레임마다 PIL 로 왔다 갔다 하면 느리므로, 글자를 모아뒀다가 한 번에 그린다.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# (최소, 최대, 이름, 설명, 색 BGR)
#
# 밀착 경계 0.65m 는 2026-07-27 에 본인이 영상을 보고 정한 값이다.
# 처음엔 통상 기준인 0.8m 를 썼는데, 실제 장면을 거리별로 늘어놓고 보니
# 0.75m 는 클린치가 아니라 아직 주고받는 근접 타격이었다.
# 나머지 두 경계는 기준표를 보고 그대로 두기로 했다.
#
# 이 값은 체격과 리치에 따라 달라진다. 다른 사람 영상에 쓰려면 다시 정해야 한다.
RANGE_BANDS = [
    (0.0, 0.65, "밀착", "클린치", (70, 70, 235)),
    (0.65, 1.3, "근거리", "훅·어퍼 거리", (60, 160, 245)),
    (1.3, 1.9, "중거리", "잽·스트레이트 거리", (90, 200, 90)),
    (1.9, 99.0, "원거리", "사거리 밖", (200, 160, 80)),
]

BAR_MAX_M = 2.6      # 막대 오른쪽 끝이 몇 m 인지 (관측 최대 2.52m)

# 두 사람이 화면에서 겹치면 검출이 두 가지로 깨진다.
#   - 한 사람에 박스가 2개 생김  -> 거리가 0에 가깝게 나옴
#   - 두 사람이 박스 1개로 합쳐짐 -> 발 위치가 누구 것인지 모호
# 둘 다 거리를 줄이는 방향으로 틀리기 때문에, 겹침이 심할수록 밀착으로 오판한다.
# 그래서 이 값 아래의 거리는 측정값으로 인정하지 않는다.
MIN_PLAUSIBLE_M = 0.30    # 서 있는 두 사람의 발 중심이 이보다 가까울 수 없다

# 신뢰도가 이보다 낮으면 구간 이름을 아예 표시하지 않는다.
# "밀착 (신뢰도 낮음)" 이라고 쓰면 사람은 '밀착' 만 읽는다. 못 잰 건 못 쟀다고 해야 한다.
CONF_SHOW_MIN = 0.5

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


def band_of(distance_m):
    """거리에 해당하는 구간을 돌려준다. 없으면 None."""
    if distance_m is None:
        return None
    for lo, hi, name, desc, color in RANGE_BANDS:
        if lo <= distance_m < hi:
            return name, desc, color
    return None


_font_cache = {}


def get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    for path in FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[size] = f
            return f
        except OSError:
            continue
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]


class TextLayer:
    """한글 글자를 모아뒀다가 프레임당 한 번만 그린다."""

    def __init__(self):
        self.items = []

    def add(self, text, xy, size=28, color=(255, 255, 255), anchor="la", outline=3):
        """color 는 BGR (OpenCV 와 맞춤). outline 은 검은 테두리 두께."""
        self.items.append((text, xy, size, color, anchor, outline))

    def render(self, bgr):
        if not self.items:
            return bgr
        img = Image.fromarray(bgr[:, :, ::-1])
        d = ImageDraw.Draw(img)
        for text, xy, size, color, anchor, outline in self.items:
            rgb = (color[2], color[1], color[0])
            d.text(xy, text, font=get_font(size), fill=rgb, anchor=anchor,
                   stroke_width=outline, stroke_fill=(0, 0, 0))
        self.items.clear()
        return np.ascontiguousarray(np.array(img)[:, :, ::-1])
