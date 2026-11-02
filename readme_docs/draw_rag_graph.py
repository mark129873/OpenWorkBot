# /// script
# requires-python = ">=3.11"
# dependencies = ["Pillow==13.0.0"]
# ///
"""Windows 下运行：uv run --script readme_docs/draw_rag_graph.py。"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


scale = 2
width, height = 1536, 2120
image = Image.new("RGB", (width * scale, height * scale), "#f8f7f4")
draw = ImageDraw.Draw(image)
regular = "C:/Windows/Fonts/msyh.ttc"
bold = "C:/Windows/Fonts/msyhbd.ttc"
fonts = {
    "title": ImageFont.truetype(bold, 22 * scale),
    "body": ImageFont.truetype(regular, 18 * scale),
    "small": ImageFont.truetype(regular, 17 * scale),
    "end": ImageFont.truetype(bold, 21 * scale),
}
colors = {
    "line": "#878783",
    "dashed": "#8b9a9b",
    "neutral": ("#f0eee9", "#77736d", "#41403d"),
    "planner": ("#efedfc", "#6655bd", "#3e347d"),
    "strategy": ("#e3f4ee", "#218167", "#145744"),
    "retrieval": ("#e6f1fa", "#3477ad", "#194f79"),
    "grade": ("#fff1d9", "#ae7826", "#704715"),
    "external": ("#f2f4f4", "#81959a", "#40575b"),
}


def xy(points):
    return [(int(x * scale), int(y * scale)) for x, y in points]


def line(points, color=None, width=2, dash=False):
    color = color or colors["line"]
    if not dash:
        draw.line(xy(points), fill=color, width=width * scale, joint="curve")
        return
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        distance = max(abs(x2-x1), abs(y2-y1))
        if not distance:
            continue
        pos = 0
        while pos < distance:
            end = min(pos + 6, distance)
            t1, t2 = pos / distance, end / distance
            draw.line(xy([(x1+(x2-x1)*t1, y1+(y2-y1)*t1),
                          (x1+(x2-x1)*t2, y1+(y2-y1)*t2)]),
                      fill=color, width=width * scale)
            pos += 11


def arrow(x, y, direction="down", color=None):
    color = color or colors["line"]
    shapes = {
        "down": [(x, y), (x-6, y-10), (x+6, y-10)],
        "right": [(x, y), (x-10, y-6), (x-10, y+6)],
        "left": [(x, y), (x+10, y-6), (x+10, y+6)],
    }
    draw.polygon(xy(shapes[direction]), fill=color)


def centered(x, y, text, font, fill):
    draw.text((x*scale, y*scale), text, font=font, fill=fill, anchor="mm")


def box(x, y, w, h, role, title, body=()):
    assert draw.textlength(title, font=fonts["title"]) <= (w-16)*scale
    assert all(draw.textlength(item, font=fonts["body"]) <= (w-16)*scale for item in body)
    fill, stroke, ink = colors[role]
    draw.rounded_rectangle((x*scale, y*scale, (x+w)*scale, (y+h)*scale),
                           radius=14*scale, fill=fill, outline=stroke, width=2*scale)
    total = 27 + len(body)*28
    top = y + (h-total)/2 + 12
    centered(x+w/2, top, title, fonts["title"], ink)
    for i, item in enumerate(body):
        centered(x+w/2, top+32+i*28, item, fonts["body"], ink)


def endpoint(x, y, w, h, title, subtitle):
    fill, stroke, ink = colors["neutral"]
    draw.rounded_rectangle((x*scale, y*scale, (x+w)*scale, (y+h)*scale),
                           radius=27*scale, fill=fill, outline=stroke, width=2*scale)
    centered(x+w/2, y+25, title, fonts["end"], ink)
    centered(x+w/2, y+50, subtitle, fonts["small"], ink)


def cylinder(x, y, w, h, title, body):
    fill, stroke, ink = colors["external"]
    ellipse = 22
    draw.rectangle((x*scale, (y+ellipse/2)*scale, (x+w)*scale, (y+h-ellipse/2)*scale), fill=fill)
    draw.ellipse((x*scale, (y+h-ellipse)*scale, (x+w)*scale, (y+h)*scale),
                 fill=fill, outline=stroke, width=2*scale)
    draw.line(xy([(x,y+ellipse/2), (x,y+h-ellipse/2)]), fill=stroke, width=2*scale)
    draw.line(xy([(x+w,y+ellipse/2), (x+w,y+h-ellipse/2)]), fill=stroke, width=2*scale)
    draw.ellipse((x*scale, y*scale, (x+w)*scale, (y+ellipse)*scale),
                 fill="#f8fafa", outline=stroke, width=2*scale)
    centered(x+w/2, y+43, title, fonts["body"], ink)
    for i, item in enumerate(body):
        centered(x+w/2, y+68+i*23, item, fonts["small"], ink)


axis = 1000
strategy_centers = [209, 581, 953, 1325]
line([(axis,114), (axis,164)]); arrow(axis,164)
line([(axis,324), (axis,385)])
line([(strategy_centers[0],385), (strategy_centers[-1],385)])
for x in strategy_centers:
    line([(x,385),(x,430)]); arrow(x,430)
    line([(x,610),(x,660)])
line([(strategy_centers[0],660), (strategy_centers[-1],660)])
line([(axis,660),(axis,725)]); arrow(axis,725)
line([(axis,905),(axis,1085)]); arrow(axis,1085)
line([(axis,1240),(axis,1390)]); arrow(axis,1390)
line([(axis,1550),(axis,1700)]); arrow(axis,1700)
line([(axis,1860),(axis,2000)]); arrow(axis,2000)
line([(1200,1470),(1515,1470),(1515,244),(1270,244)])
arrow(1270,244,"left")
for x, y in [(150,870),(395,870),(640,880)]:
    line([(x,y),(x,930)], color=colors["dashed"], dash=True)
line([(150,930),(780,930),(780,850),(800,850)], color=colors["dashed"], dash=True)
arrow(800,850,"right",colors["dashed"])
line([(750,1163),(800,1163)], color=colors["dashed"], dash=True)
arrow(800,1163,"right",colors["dashed"])

endpoint(895,44,210,70,"RAG START","开始检索")
box(730,164,540,160,"planner","plan · 检索规划",
    ("分析问题和证据缺口", "选择一种或多种查询策略", "分配每轮查询预算"))
box(34,430,350,180,"strategy","rewrite · 问题重写",
    ("整理成清晰的检索问题", "适合检索时可保留原问题", "最多生成 W 条"))
box(406,430,350,180,"strategy","multi_query · 多查询",
    ("围绕同一问题", "生成多种不同检索表达", "保留原意和条件"))
box(778,430,350,180,"strategy","expand · 查询扩展",
    ("补充同义词和相关术语", "扩大关键词覆盖", "不引入新事实"))
box(1150,430,350,180,"strategy","subquery · 问题拆分",
    ("按所需证据拆分问题", "生成多个独立查询", "保留必要条件"))

cylinder(40,760,220,110,"本地 Embedding",("将查询转成向量",))
cylinder(285,760,220,110,"SQLite 目录",("提供有效文档版本",))
cylinder(530,750,220,130,"Milvus Cloud",("Dense + BM25", "RRF 融合 Top F"))
box(800,725,400,180,"retrieval","retrieve · 混合检索",
    ("汇总、去重查询，每轮最多 Q 条", "并行语义与关键词检索", "合并候选，最多保留 C 块"))

cylinder(530,1105,220,115,"本地 Reranker",("计算相关度",))
box(800,1085,400,155,"retrieval","rerank_stage · 证据重排",
    ("使用原问题对候选重新排序", "保留最多 E 块证据"))

box(800,1390,400,160,"grade","grade · 证据评估",
    ("判断证据是否足够", "识别缺失依据", "判断是否可以补检"))
draw.rounded_rectangle((1215*scale,1332*scale,1492*scale,1400*scale),
                       radius=10*scale, fill="#f8f7f4")
centered(1353.5,1350,"证据不足且可以补检",fonts["small"],"#704715")
centered(1353.5,1374,"最多补检 T 轮",fonts["small"],"#704715")

box(800,1700,400,160,"planner","answer · 生成答案",
    ("根据证据回答并附文件名", "证据不足时说明限制", "返回 success 或 insufficient"))
endpoint(895,2000,210,70,"RAG END","返回答案与来源")

output = Path(__file__).resolve().with_name("rag_langgraph.png")
image.save(output, "PNG", optimize=True)
print(output)
