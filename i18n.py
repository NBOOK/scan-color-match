"""User-interface language: English (source strings) or Simplified Chinese.

All interface text is written in English in the code. In Chinese mode it is translated
just before it is shown: exact strings through ``ZH``, messages that contain values
(file names, counts, engine errors) through ``PATTERNS``. Composite messages joined by
" · ", "; " or line breaks are translated piece by piece. Untranslated text is shown
unchanged, so nothing is ever lost. Engine logs and report.json stay in English.
"""

from __future__ import annotations

import re

LANGUAGES = {"en": "English", "zh": "中文"}
_language = "en"


def set_language(code: str) -> None:
    global _language
    _language = code if code in LANGUAGES else "en"


def language() -> str:
    return _language


ZH = {
    # Window, menus, dialogs
    "Scan Color Match": "Scan Color Match 扫图校色",
    "File": "文件",
    "Edit": "编辑",
    "Help": "帮助",
    "Language": "语言",
    "New Project...": "新建项目...",
    "Open Project...": "打开项目...",
    "Clear Project...": "清空项目...",
    "Clear Project": "清空项目",
    "Add Scans...": "添加扫图...",
    "Add References...": "添加参考图...",
    "Batch Import (ordered pairs)...": "批量导入（按顺序配对）...",
    "Export Current...": "导出当前...",
    "Export All...": "全部导出...",
    "Include Comparison Previews": "同时导出对比预览图",
    "Include JSON Reports": "同时导出 JSON 报告",
    "Exit": "退出",
    "Undo": "撤销",
    "Redo Mask Edit": "重做遮罩编辑",
    "Restore Removed Scan/Reference": "恢复刚移除的扫图/参考图",
    "Quick Start...": "快速上手...",
    "Quick Start": "快速上手",
    "Batch Import": "批量导入",
    "Drop files into each list. Rows are paired by order, not by filename. Drag rows to reorder. "
    "Extra scans have no reference.":
        "把文件拖入两侧列表。按行号配对，不看文件名；拖动行可调整顺序。多出的扫图没有参考图。",
    "Scans": "扫图",
    "References": "参考图",
    "Add Scans": "添加扫图",
    "Add References": "添加参考图",
    "Remove selected": "移除选中项",
    "Import Pairs": "导入配对",
    "Cancel": "取消",
    "Choose scans": "选择扫图",
    "Choose references": "选择参考图",
    "Add scans": "添加扫图",
    "Choose an empty project folder": "选择一个空文件夹作为项目",
    "Choose a project folder": "选择项目文件夹",
    "Choose export folder": "选择导出文件夹",
    "Rename Layer": "重命名图层",
    "Layer name": "图层名称",
    "Clear all scans, references, layers and generated cache from this project?":
        "清空此项目中的所有扫图、参考图、图层和生成的缓存？",
    "Original images and exported files will remain untouched.": "原始图片和已导出的文件不受影响。",
    "Original images and exported PNGs will remain untouched.": "原始图片和已导出的 PNG 不受影响。",
    "Restart Scan Color Match to use the new language.": "重新启动程序后使用新的界面语言。",
    # Main window
    "No project": "没有项目",
    "Scan / Result — drop scan to replace": "扫图 / 结果 —— 拖入扫图可替换",
    "Reference — drop image to replace": "参考图 —— 拖入图片可替换",
    "Drop a scan into the left preview to begin. Hold Space to compare the original.":
        "把扫图拖到左侧预览开始。按住空格键对比原图。",
    "Drop a scan into the left preview to begin.": "把扫图拖到左侧预览开始。",
    "Scan file is missing. Drop its new location into the left preview.":
        "扫图文件不见了。把它的新位置拖到左侧预览即可重新关联。",
    "Mean CIE76 Lab color difference on the selected layer's mask. Off and On keep all other "
    "layers unchanged and include the current mask, strength and line protection.":
        "所选图层遮罩内的平均 CIE76 Lab 色差。“关/开”对比时其他图层保持不变，并包含当前遮罩、强度和暗线保护。",
    "Mean ΔE76 on the global calibration sample; lower is better.": "全局校色样本上的平均 ΔE76，越小越好。",
    "Final ΔE76: updating…": "最终 ΔE76：更新中…",
    "Mask updating…": "遮罩 更新中…",
    "Final updating…": "最终 更新中…",
    "ΔE76: out of date": "ΔE76：已过期",
    "Mean ΔE76 against the aligned reference, lower is better": "与对齐后参考图的平均 ΔE76，越小越好",
    "Mask: the selected layer off → on, inside its mask": "遮罩：所选图层关 → 开，在其遮罩内测量",
    "Mask ΔE76: updating…": "遮罩 ΔE76：更新中…",
    "Final ΔE76: measuring…": "最终 ΔE76：测量中…",
    "Mean ΔE76 against the aligned reference on the calibration sample, lower is better":
        "在校色样本上与对齐后参考图的平均 ΔE76，越小越好",
    "Global: original scan → global LUT and spatial correction":
        "全局：原始扫图 → 全局 LUT 与空间修正",
    "Final: original scan → the result shown, with the global switch and all local layers":
        "最终：原始扫图 → 当前显示的结果（包含全局开关和所有局部图层）",
    "Use reference": "使用此参考图",
    "Disabled: not used for color matching. Check to use it again": "已停用：不参与校色。勾选即可重新启用",
    "Sampling Mask": "取样遮罩",
    "Paint on the reference preview which reference colors may be used as evidence; saved in "
    "original reference coordinates. The LUT still applies to the full scan.":
        "在参考图预览上涂出哪些参考颜色可以作为校色依据；按参考原图坐标保存。LUT 仍作用于整张扫图。",
    "Exclude painted": "排除选区",
    "Exclusion": "排除选区",
    "Inclusion": "仅用选区",
    "Painted pixels (red) are never used: watermarks, text, content that does not match the scan":
        "涂上的像素（红色）不会被使用：水印、文字、与扫图不对应的内容",
    "Use only painted": "仅用选区",
    "Only painted pixels (green) are used: the part of the reference that matches the scan":
        "只使用涂上的像素（绿色）：参考图中与扫图对应的部分",
    "Edit Mask": "编辑遮罩",
    "Clear Mask": "清除遮罩",
    "Show mask": "显示遮罩",
    "Rectangle": "矩形",
    "Polygon": "多边形",
    "Brush": "画笔",
    "Eraser": "橡皮擦",
    "Brush diameter": "笔刷直径",
    "Red = excluded": "红色 = 排除",
    "Green = only area used": "绿色 = 仅用此区域",
    "Ctrl+Z undo": "Ctrl+Z 撤销",
    "Ctrl+Y redo": "Ctrl+Y 重做",
    "Esc exit": "Esc 退出",
    "LUT Mode": "LUT 模式",
    "Shared": "共享",
    "Own LUT": "独立 LUT",
    "Shared LUT": "共享 LUT",
    "Train shared too": "同时参与共享训练",
    "Shared + joint training": "共享 + 联合训练",
    "Learn a LUT only from this scan's references. It does not help other scans":
        "只用这张扫图自己的参考图训练 LUT，不参与其他扫图的训练",
    "Help train one shared LUT together with the other shared scans, and use that shared LUT. "
    "Best when this scan's references are few or partial":
        "和其他扫图一起训练一张共享 LUT，并使用这张共享 LUT。适合本图参考图较少或只对应一部分的情况",
    "Use a LUT learned from this scan's references, and also help train the shared LUT for "
    "other scans. Recommended":
        "使用由本图参考图训练的独立 LUT，同时帮助训练供其他扫图使用的共享 LUT。推荐",
    "Use the shared LUT learned from the scans that have references": "使用由有参考图的扫图训练出的共享 LUT",
    "Own + joint training": "独立 + 联合训练",
    "Own + joint LUT": "独立+联合 LUT",
    "Own LUT: learn only from this scan's references": "独立 LUT：只用这张扫图自己的参考图训练",
    "Shared + joint training: help train the shared LUT and use it": "共享 + 联合训练：参与训练共享 LUT 并使用它",
    "Own + joint training: use its own LUT and also help train the shared LUT":
        "独立 + 联合训练：使用自己的独立 LUT，同时帮助训练共享 LUT",
    "Include this Own LUT scan in shared LUT training": "让这张使用独立 LUT 的扫图也参与共享 LUT 的训练",
    "Layers": "图层",
    "Select a reference, click +, paint or use polygon selection, then Refine All. Click a "
    "thumbnail to view the result or mask; drag layers to reorder.":
        "选中参考图，点 +，用画笔或多边形圈出区域，再点“全部精修”。点缩略图可查看结果或遮罩；拖动图层可调整顺序。",
    "Tool": "工具",
    "Diameter": "直径",
    "Hardness": "硬度",
    "Pan": "平移",
    "Protect dark lines": "保护暗线",
    "Reduce the selected layer's color correction on very dark lines": "减弱所选图层对很暗线条的颜色修正",
    "Mask View": "查看遮罩",
    "Show the selected layer mask alone in black and white": "以黑白单独显示所选图层的遮罩",
    "Lasso": "套索",
    "Polygon selection (L): click vertices, then click the first point to close":
        "多边形选区（L）：逐点点击顶点，最后点击起点闭合",
    "Paint": "画笔",
    "Brush (B)": "画笔（B）",
    "Erase": "橡皮擦",
    "Eraser (E)": "橡皮擦（E）",
    "X switches between the eraser and the previous tool": "X 在橡皮擦和之前的工具间切换",
    "Refine All": "全部精修",
    "Re-fit changed layers bottom-to-top, then automatically choose their strengths":
        "从下到上重新拟合有改动的图层，并自动选择强度",
    "Run Color Match": "运行校色",
    "Export...": "导出...",
    "Ready": "就绪",
    'Fit target': '拟合目标',
    'Preparing layers...': '正在准备图层...',
    'Saving the result...': '正在保存结果...',
    'Refine All finished': '全部精修完成',
    'Updating local layers...': '正在更新局部图层...',
    'Local layers updated': '局部图层已更新',
    'Saving the global result...': '正在保存全局结果...',
    'Global result updated': '全局结果已更新',
    'Reused the rendered global result': '已复用之前生成的全局结果',
    'Rectangle (R)': '矩形（R）',
    'Polygon (L): click the first point to close': '多边形（L）：点击起点闭合',
    'Eraser (E) · X switches back': '橡皮擦（E） · X 切回之前的工具',
    'click again to leave the tool': '再次点击退出工具',
    'Ctrl+Z undo · Ctrl+Y redo · Esc exit': 'Ctrl+Z 撤销 · Ctrl+Y 重做 · Esc 退出',
    'Feather': '羽化',
    'Polygon edge feather: the soft transition on each side of the outline, in image pixels. 0 is a hard edge': '多边形边缘羽化：轮廓线内外两侧的柔和过渡宽度（图像像素），0 为硬边',
    'Left bar: blue Own + joint, purple Own LUT, teal Shared + joint': '左侧色条：蓝色 = 独立 + 联合训练，紫色 = 独立 LUT，青色 = 共享 + 联合训练',
    'RGB': 'RGB',
    'What the global LUT is fitted to. Validation and the ΔE readouts always use ΔE76.': '全局 LUT 按哪种误差来拟合。验收和界面上的 ΔE 读数始终用 ΔE76。',
    'ΔE76: fits the perceptual color difference directly. Better results, a few seconds slower per pass.': 'ΔE76：直接按人眼色差拟合。效果更好，每轮多几秒。',
    'RGB: fits the RGB error. Fastest, slightly less accurate.': 'RGB：按 RGB 数值误差拟合。最快，略不准。',
    'Passes': '迭代次数',
    'Global passes for scans with their own LUT: each pass corrects what the previous one left. Every further pass gains less, costs a few seconds per scan, and fits misalignment and noise in the reference more closely (overfitting). A pass that fails held-out validation is not used.': '使用独立 LUT 的扫图做几轮全局校正：每一轮修正上一轮剩下的误差。轮数越多每轮收益越小，每张图每轮多花几秒，而且会越来越多地去贴合参考图里的错位和噪点（过拟合）。没通过留出验证的那一轮不会被采用。',
    "The global layer's strength then picks the point with the lowest ΔE76; you can change it afterwards.": '之后全局图层的强度会自动取 ΔE76 最低的位置，也可以手动修改。',
    'Checkbox: apply the full-scan LUT and spatial correction': '复选框：应用整页 LUT 和空间校正',
    'double-click to rename': '双击重命名',
    'Percentage: global strength. 100% is one pass, each further 100% one more pass, blended in between. Unless you set it, it is chosen automatically: the lowest ΔE76. Click to edit, wheel ±10%, right-click for automatic': '百分比：全局强度。100% 是一轮，每多 100% 多一轮，中间按比例插值。没有手动设置时自动取 ΔE76 最低的位置。点击编辑，滚轮 ±10%，右键可恢复自动',
    'Automatic strength (lowest ΔE76)': '自动强度（ΔE76 最低）',
    'Fit settings changed. Run Color Match to apply them': '拟合设置已更改，运行校色后生效',
    'Fitting the shared LUT': '正在拟合共享 LUT',
    'Fitted global passes': '全局迭代拟合完成',
    'Fitting more global passes...': '正在拟合更多全局迭代...',
    'Rendering the global result...': '正在生成全局结果...',
    'Only scans with their own LUT can get more passes': '只有使用独立 LUT 的扫图能追加迭代',
    "New layer": "新建图层",
    "＋  New layer": "＋  新建图层",
    "global": "全局",
    "Rename": "重命名",
    "Delete layer": "删除图层",
    "Correction strength...": "修正强度...",
    "Show mask overlay": "显示遮罩叠加",
    "View layer mask": "查看图层遮罩",
    " (hidden)": "（已隐藏）",
    "Polygon selection has no brush diameter": "多边形选区没有笔刷直径",
    "Toggle the full-scan LUT and spatial correction; double-click to rename":
        "开关整张扫图的 LUT 和空间修正；双击可重命名",
    "Checkbox: apply layer": "复选框：应用此图层",
    "Eye: show overlay": "眼睛：显示遮罩叠加",
    "Percentage: click to edit, wheel ±10%": "百分比：点击可编辑，滚轮 ±10%",
    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.webp)": "图片 (*.png *.jpg *.jpeg *.tif *.tiff *.webp)",
    # Scan and reference states
    "Missing source": "源文件丢失",
    "Previous result": "上次的结果",
    "rerun needed": "需要重新运行",
    "new match failed": "新一次匹配失败",
    "Out of date": "已过期",
    "Pending": "待处理",
    "Applying LUT…": "正在套用 LUT…",
    "Applying the existing LUT to the changed scan(s)...": "正在为改动的扫图套用已有的 LUT...",
    "Run Color Match first": "请先运行校色",
    "Scans with references need Run Color Match": "有参考图的扫图需要运行校色",
    "Needs Refine All": "需要全部精修",
    "Updating local layers": "正在更新局部图层",
    "Reference failed": "参考图失败",
    "Reference attached": "已挂参考图",
    "Needs shared LUT": "需要共享 LUT",
    "Empty image pair": "空的图片对",
    "pending": "待处理",
    "aligned": "已对齐",
    "failed": "失败",
    "missing": "丢失",
    "exclusions": "排除遮罩",
    "use only mask": "仅用遮罩",
    "reference sampling mask changed": "参考取样遮罩已修改",
    "Choose a reference or review the scan": "选择一张参考图，或检查扫图",
    "ΔE76: out of date": "ΔE76：已过期",
    "Selected layer ΔE76: use Refine All to measure its effect": "所选图层 ΔE76：请先“全部精修”再测量效果",
    "Selected layer ΔE76: updating actual Off / On result…": "所选图层 ΔE76：正在更新实际的关/开结果…",
    "Selected layer ΔE76: measuring actual Off / On result…": "所选图层 ΔE76：正在测量实际的关/开结果…",
    "Selected layer ΔE76: no measurable aligned mask pixels": "所选图层 ΔE76：遮罩内没有可测量的已对齐像素",
    "Off and On keep all other layers unchanged and include the current mask, strength and line "
    "protection. Lower ΔE76 is better.":
        "关/开对比时其他图层保持不变，并包含当前遮罩、强度和暗线保护。ΔE76 越小越好。",
    "enabled": "已启用",
    "disabled": "已停用",
    # Status messages
    "Scan replaced. Run Color Match to update the result": "扫图已替换。请运行校色以更新结果",
    "Reference replaced. Run Color Match to update the result": "参考图已替换。请运行校色以更新结果",
    "Updating saved local-layer settings...": "正在更新已保存的局部图层设置...",
    "Reference sampling mask saved. Run Color Match to update the LUT and corrections":
        "参考取样遮罩已保存。请运行校色以更新 LUT 和修正",
    "Paint the reference colors to use only (green). Edits save automatically; rerun Color Match when done":
        "涂出只使用的参考颜色（绿色）。修改会自动保存；完成后重新运行校色",
    "Mark untrusted reference colors in red. Edits save automatically; rerun Color Match when done":
        "用红色标出不可信的参考颜色。修改会自动保存；完成后重新运行校色",
    "Use only painted: paint the matching part of the reference": "仅用选区：在参考图上涂出与扫图对应的部分",
    "Exclude painted: paint untrusted reference colors": "排除选区：涂出不可信的参考颜色",
    "Clear Mask to reset": "可用“清除遮罩”重置",
    "Global correction enabled": "已启用全局校正",
    "Global correction disabled": "已停用全局校正",
    "Updating local layer preview...": "正在更新局部图层预览...",
    "Local layer preview updated": "局部图层预览已更新",
    "Local layer visibility updated": "局部图层可见性已更新",
    "Painted layer needs Refine All to fit its color correction": "涂好的图层需要“全部精修”来拟合颜色修正",
    "Preview updated; rendering full-resolution result...": "预览已更新；正在生成全分辨率结果...",
    "Layer mask could not be saved; full-resolution result was not updated":
        "图层遮罩保存失败；全分辨率结果没有更新",
    "Mask changed. Click Refine All to fit its color correction": "遮罩已修改。请点“全部精修”拟合颜色修正",
    "Cancelling after the current step...": "当前步骤结束后取消...",
    "Cancelled; the previous result is unchanged": "已取消，上一次的结果保持不变",
    "Refine All: cached result reused": "全部精修：沿用了缓存的结果",
    "Refine All: corrections and strengths updated": "全部精修：修正和强度已更新",
    "Closing after the current operation stops...": "当前操作停止后关闭...",
    "Project cleared": "项目已清空",
    # Errors shown by the window
    "Create or open a project first": "请先新建或打开一个项目",
    "Select or add a scan card first": "请先选择或添加一张扫图",
    "Select a scan card first": "请先选择一张扫图",
    "Add at least one scan": "请至少添加一张扫图",
    "There are more references than scans. Remove extra references or add scans.":
        "参考图比扫图多。请移除多余的参考图或添加扫图。",
    "Wait for the current operation to finish before clearing the project": "请等当前操作完成后再清空项目",
    "Choose a scan and reference before adding a layer": "添加图层前请先选择扫图和参考图",
    "Choose a scan before editing a layer": "编辑图层前请先选择扫图",
    "Add a reference before editing a layer": "编辑图层前请先添加参考图",
    "Choose a scan and reference first": "请先选择扫图和参考图",
    "Another operation is already running": "已有其他操作正在运行",
    "The project changed during processing. Run again": "处理期间项目发生了变化。请重新运行",
    "Paint at least one correction layer first": "请先至少涂一个修正图层",
    "Could not add:": "无法添加：",
    # Engine progress and errors (engine and reports stay in English)
    "Fitting LUTs": "正在拟合 LUT",
    "Fitting individual LUTs": "正在拟合独立 LUT",
    "Preparing full-size correction": "正在准备全尺寸校正",
    "Correcting full-size scans": "正在校正全尺寸扫图",
    "Color matching complete": "校色完成",
    "Refining correction layers": "正在精修局部图层",
    "Auto Strength complete": "自动强度完成",
    "Color matching was cancelled": "校色已取消",
    "Too little aligned reference coverage": "参考图对齐后的覆盖范围太小",
    "Too few reliable flat-color samples": "可靠的平坦颜色样本太少",
    "Too few aligned pixels inside the reference sampling mask": "参考取样遮罩内的已对齐像素太少",
    "Too few reliable color samples inside the reference sampling mask": "参考取样遮罩内的可靠颜色样本太少",
    "enlarge the included area or reduce the exclusions": "请扩大使用区域或缩小排除区域",
    "Add at least one scan before running": "运行前请至少添加一张扫图",
    "Shared LUT failed validation": "共享 LUT 未通过验证",
    "try Own LUT": "可改用独立 LUT",
    "Reference is not attached to this scan": "这张参考图不属于当前扫图",
    "Select at least three polygon points": "多边形至少需要三个点",
    "The polygon must enclose an area": "多边形必须围出一块区域",
    "The polygon is outside the image": "多边形在图像范围之外",
    "Paint a larger area inside the aligned reference": "请在已对齐的参考图范围内涂更大的区域",
    "Not enough reliable color pixels in the painted area": "涂抹区域内可靠的颜色像素不足",
    "Not enough validation pixels in the painted area": "涂抹区域内用于验证的像素不足",
    "Local correction did not improve held-out pixels": "局部修正没有改善留出验证像素",
    "Run Color Match before refining this scan": "请先运行校色，再精修这张扫图",
    "This scan has no calibrated result": "这张扫图还没有校色结果",
    "Refine All reused the current result": "全部精修：沿用了当前结果",
    "Run Color Match before updating local layers": "请先运行校色，再更新局部图层",
    "Use Refine All before Auto Strength": "使用自动强度前请先“全部精修”",
    "The selected layer has no aligned reference": "所选图层没有已对齐的参考图",
    "Paint the selected layer before using Auto Strength": "使用自动强度前请先涂抹所选图层",
    "Not enough aligned pixels inside the selected mask": "所选遮罩内的已对齐像素不足",
    "Not enough aligned pixels allowed by the reference sampling mask inside the selected mask":
        "所选遮罩内被参考取样遮罩允许的已对齐像素不足",
    "The result is out of date. Run matching or refine the painted layers first":
        "结果已过期。请先运行校色或精修涂好的图层",
    "Result size differs from the scan": "结果尺寸与扫图不一致",
    "16-bit PNG is not supported in this version": "此版本不支持 16 位 PNG",
    "16-bit images are not supported in this version": "此版本不支持 16 位图像",
    "16-bit TIFF is not supported in this version": "此版本不支持 16 位 TIFF",
    "Reference exclusion mask must be inside the project": "参考遮罩必须位于项目内",
    "Reference exclusion mask content changed": "参考遮罩文件内容被修改",
    "Reference exclusion mask size differs from the reference": "参考遮罩尺寸与参考图不一致",
    "Reference exclusion mask must contain only black and white": "参考遮罩只能包含黑白两色",
    "Reference exclusion mask must be a grayscale PNG": "参考遮罩必须是灰度 PNG",
    "Own LUT needs an aligned reference": "独立 LUT 需要至少一张成功对齐的参考图",
    "Shared LUT required": "需要共享 LUT",
    "The followed scan has no validated Own LUT": "所沿用的扫图没有通过验证的独立 LUT",
    "Scans without references use the shared LUT, or the Own LUT of another scan "
    "(for example a back cover following its front cover)":
        "没有参考图的扫图使用共享 LUT，或沿用另一张扫图的独立 LUT（例如封底跟随封面）",
    "Shared: learn one LUT from all shared scans": "共享：用所有共享扫图共同训练一张 LUT",
    "Own LUT: learn from this scan's references": "独立 LUT：只用这张扫图自己的参考图训练",
    "Shared LUT failed restricted color sampling validation": "共享 LUT 未通过受限取样的验证",
    "Shared LUT failed held-out validation": "共享 LUT 未通过留出验证",
    "Restricted color sampling did not improve held-out regions": "受限取样没有改善留出区域",
    "Own LUT failed held-out validation": "独立 LUT 未通过留出验证",
    "Perceptual fitting requires scipy": "感知拟合需要 scipy",
    "install requirements.txt": "请安装 requirements.txt",
    "Process isolation unavailable": "无法使用进程隔离",
    "continuing with memory-limited workers": "改用受内存限制的并行任务继续",
    "the previous result is unchanged": "上一次的结果保持不变",
    "Calibration process disconnected": "校色进程连接中断",
    "Unsupported project format": "不支持的项目格式",
    "Operation cancelled": "操作已取消",
}

# (regex, Chinese template). Groups are inserted as {0}, {1}, ...; a group written as
# {0!t} is itself translated first (for nested engine messages).
PATTERNS = [
    (r"Refining (.+)", "正在精修 {0}"),
    (r"Choosing the strength of (.+)", "正在选择 {0} 的强度"),
    (r"Updated (.+)", "已更新 {0}"),
    (r"Pass (\d+) failed held-out validation \(used only to blend towards\): at most (\d+)%", "第 {0} 轮未通过留出验证（仅用于插值）：最高 {1}%"),
    (r"pass (\d+) failed held-out validation and is used only to blend towards", "第 {0} 轮未通过留出验证，仅用于插值"),
    (r"Polygon edge feather (\d+) px", "多边形边缘羽化 {0} px"),
    (r"Feather:\s+(\d+) px", "羽化：{0} px"),
    (r"Global (\d+)%", "全局 {0}%"),
    (r"(\d+) layer\(s\)", "{0} 个图层"),
    (r"Polygon edge hardness (\d+)%", "多边形边缘硬度 {0}%"),
    (r'Global strength (\d+)% \(automatic\)', '全局强度 {0}%（自动）'),
    (r'Global strength (\d+)%', '全局强度 {0}%'),
    (r'only (\d+) pass\(es\) passed validation', '只有 {0} 轮通过验证'),
    (r'about (\d+) s left', '约剩 {0} 秒'),
    (r'about (\d+) min left', '约剩 {0} 分钟'),
    (r'Fitted pass (\d+) of (.+)', '已完成 {1} 的第 {0} 轮拟合'),
    (r'Fitted pass (\d+)', '已完成第 {0} 轮拟合'),
    (r'Aligned a reference of (.+)', '已对齐 {0} 的一张参考图'),
    (r"Added (\d+) scan\(s\)", "已添加 {0} 张扫图"),
    (r"Added (\d+) reference\(s\)", "已添加 {0} 张参考图"),
    (r"Imported (\d+) image pair\(s\)", "已导入 {0} 组配对图片"),
    (r"(\d+) reference\(s\)", "{0} 张参考图"),
    (r"Brush diameter (\d+) px", "笔刷直径 {0} px"),
    (r"hardness (\d+)%", "硬度 {0}%"),
    (r"Diameter:\s+(\d+) px", "直径：{0} px"),
    (r"Hardness:\s+(\d+)%", "硬度：{0}%"),
    (r"Color matching complete: (\d+) scan\(s\) ready", "校色完成：{0} 张扫图就绪"),
    (r"Exported (\d+) image\(s\); (\d+) skipped", "已导出 {0} 张图片，跳过 {1} 张"),
    (r"Exported all (\d+) image\(s\)\. Clear this project now\?", "已全部导出 {0} 张图片。现在清空这个项目吗？"),
    (r"Ref: (\d+) inliers", "参考图：{0} 个内点"),
    (r"ΔE76: (.+)", "ΔE76：{0!t}"),
    (r"Mask ([\d.]+) → ([\d.]+)", "遮罩 {0} → {1}"),
    (r"Global ([\d.]+) → ([\d.]+)", "全局 {0} → {1}"),
    (r"Final ([\d.]+) → ([\d.]+)", "最终 {0} → {1}"),
    (r"Global ([\d.]+)", "全局 {0}"),
    (r"Global ΔE76: ([\d.]+) → ([\d.]+)", "全局 ΔE76：{0} → {1}"),
    (r"Final ΔE76: ([\d.]+) → ([\d.]+)", "最终 ΔE76：{0} → {1}"),
    (r"Global ΔE76: ([\d.]+)", "全局 ΔE76：{0}"),
    (r"Mask ΔE76: Off ([\d.]+) → On ([\d.]+)\s+\(Δ ([+\-\d.]+); (\w+)\)", "遮罩 ΔE76：关 {0} → 开 {1}（Δ {2}；{3!t}）"),
    (r"Measured on ([\d,]+) aligned pixels inside this mask\. (.+)", "在此遮罩内 {0} 个已对齐像素上测量。{1!t}"),
    (r"Selected layer ΔE76: unavailable \((.+)\)", "所选图层 ΔE76：无法测量（{0!t}）"),
    (r"([\d,]+) reliable sample pixels at working resolution", "工作分辨率下有 {0} 个可靠样本像素"),
    (r"Updating (.+) for export", "正在为导出更新 {0}"),
    (r"Exporting (.+)", "正在导出 {0}"),
    (r"Could not save layer mask: (.+)", "无法保存图层遮罩：{0!t}"),
    (r"Could not open the default project: (.+)", "无法打开默认项目：{0!t}"),
    (r"Could not fully clear project: (.+)", "无法完全清空项目：{0!t}"),
    (r"(.+) \((pending|aligned|failed|missing)\)", "{0}（{1!t}）"),
    (r"(.+) \((pending|aligned|failed) · (exclusions|use only mask)\)", "{0}（{1!t} · {2!t}）"),
    (r"Uses #(\d+) LUT", "沿用 #{0} 的 LUT"),
    (r"Use the Own LUT of scan #(\d+), for example a back cover following its front cover",
     "沿用第 {0} 张扫图的独立 LUT，例如封底沿用封面"),
    (r"Use #(\d+) (.*) LUT", "沿用 #{0} {1} 的 LUT"),
    (r"(Shared|Own) LUT", "{0!t} LUT"),
    (r"Correction (\d+)", "修正 {0}"),
    # Engine progress
    (r"Aligning with up to (\d+) workers", "正在对齐，最多 {0} 个并行任务"),
    (r"([\d.]+) GB image-memory budget", "图像内存预算 {0} GB"),
    (r"Aligned (.+)", "已对齐 {0}"),
    (r"Correcting with (\d+) scan process\(es\) × (\d+) shared-memory band process\(es\)",
     "正在校正：{0} 个扫图进程 × {1} 个共享内存分块进程"),
    (r"Corrected (.+)", "已校正 {0}"),
    (r"Retrying (.+) with perceptual fitting", "正在用感知拟合重试 {0}"),
    (r"Reused (.+)", "已沿用 {0}"),
    (r"Refined (.+)", "已精修 {0}"),
    (r"Testing strength (\d+)%", "正在测试强度 {0}%"),
    (r"Updating (.+)", "正在更新 {0}"),
    # Engine errors
    (r"Unsupported format: (.+)", "不支持的格式：{0}"),
    (r"Unknown reference mask mode: (.+)", "未知的参考遮罩模式：{0}"),
    (r"Invalid reference exclusion mask: (.+)", "参考遮罩无效：{0!t}"),
    (r"Weak alignment: (\d+)/(\d+) inliers", "对齐不可靠：{0}/{1} 个内点"),
    (r"Cross-page correction increased held-out color error\. Use Own LUT or references with "
     r"consistent colors\. Median page Delta E: ([\d.]+) -> ([\d.]+)",
     "跨页校正反而增大了留出样本的色差。请改用独立 LUT，或使用颜色关系一致的参考图。页面色差中位数：{0} → {1}"),
    (r"Not enough aligned pixels inside (.+)'s mask", "{0} 的遮罩内已对齐像素不足"),
    (r"Mask size changed for (.+)", "{0} 的遮罩尺寸变了"),
    (r"Paint an area for (.+)", "请为 {0} 涂一块区域"),
    (r"Reference is missing for (.+)", "{0} 的参考图丢失"),
    (r"Reference did not align for (.+)", "{0} 的参考图没有对齐"),
    (r"(.+) needs Refine All before it can be shown", "{0} 需要先“全部精修”才能显示"),
    (r"(.+) needs Refine All before Auto Strength", "{0} 需要先“全部精修”才能使用自动强度"),
    (r"(.+) after the reference sampling mask", "{0!t}（受参考取样遮罩限制）"),
    (r"adjust the sampling mask or enlarge the painted area", "请调整取样遮罩或扩大涂抹区域"),
    (r"Correction process exited unexpectedly \(code (-?\d+)\)", "校正进程意外退出（代码 {0}）"),
    (r"Parallel image band exited unexpectedly \(code (-?\d+)\)", "并行分块进程意外退出（代码 {0}）"),
    (r"Calibration process exited unexpectedly \(code (-?\d+)\)", "校色进程意外退出（代码 {0}）"),
    (r"Calibration process exited with code (-?\d+)", "校色进程退出，代码 {0}"),
    (r"Color matching core is missing: (.+)", "缺少校色核心文件：{0}"),
    (r"A project already exists in (.+)", "{0} 中已经有一个项目"),
]
_PATTERNS = [(re.compile(rf"^{pattern}$", re.S), template) for pattern, template in PATTERNS]
_FIELD = re.compile(r"\{(\d+)(!t)?\}")


def _whole(text: str) -> str | None:
    if text in ZH:
        return ZH[text]
    for pattern, template in _PATTERNS:
        match = pattern.match(text)
        if match:
            groups = match.groups()
            return _FIELD.sub(lambda field: tr(groups[int(field.group(1))]) if field.group(2)
                              else groups[int(field.group(1))], template)
    return None


def tr(text):
    """Translate interface text for the current language (unchanged in English)."""
    if _language == "en" or not isinstance(text, str) or not text.strip():
        return text
    translated = _whole(text)
    if translated is not None:
        return translated
    stripped = text.strip()
    if stripped != text:
        translated = _whole(stripped)
        if translated is not None:
            return text.replace(stripped, translated)
    # Composite messages: translate each line / clause / segment on its own.
    for separator, joiner in (("\n", "\n"), (" | ", " | "), ("; ", "；"), (" · ", " · "), (". ", "。")):
        if separator in text:
            parts = text.split(separator)
            converted = [tr(part) for part in parts]
            if converted != parts:
                return joiner.join(converted)
    return text


QUICK_START_ZH = (
    "<h3>Scan Color Match 扫图校色</h3>"
    "<p><b>1.</b> 程序会自动打开默认项目，也可以新建或打开其他项目。项目自动保存，不会改动原始图片。</p>"
    "<p><b>2.</b> 把一张或多张扫图拖到左侧列表。选中一张扫图，再把它的参考图拖到“参考图”列表，文件名不需要一致。"
    "向预览拖入单张图片会替换当前选中的图片；没有选中时则添加。</p>"
    "<p><b>3.</b> 选择“共享”或“独立 LUT”，然后点“运行校色”。没有参考图的扫图使用共享 LUT。"
    "如果参考图只有一部分和扫图对应，在参考图预览上点“编辑遮罩”：排除水印或对不上的内容，"
    "或切换到“仅用选区”并涂出对应的部分。遮罩只限制训练样本，LUT 仍然作用于整张扫图。"
    "如果对齐失败，先把参考图裁切到与扫图对应的部分再添加。</p>"
    "<p><b>4.</b> 局部修正：选中参考图，用画笔涂抹或用多边形圈出区域，然后点“全部精修”。"
    "程序会从下到上拟合有改动的图层并自动选择强度。用每个图层的复选框开关它。</p>"
    "<p><b>5.</b> 完成后导出。导出使用当前可见的图层和遮罩。</p>"
    "<p><b>快捷键：</b>B 画笔 · E 橡皮擦 · X 在橡皮擦和之前的工具间切换 · L 多边形 · Esc 退出工具 · "
    "按住空格看原图 · 中键拖动平移 · 滚轮缩放。Alt + 右键拖动：横向调直径，纵向调硬度。</p>"
)
