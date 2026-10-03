"""
generate_data.py

生成配对数据: (光阑, 衍射图样)

物理模型: 远场夫琅禾费衍射
    I(u,v) = |FFT{a(x,y)}|²

后处理 (必须和 train.py 的 FraunhoferLayer 完全一致):
    1. FFT + fftshift
    2. |.|²
    3. 归一化到 [0, 1]
    4. log 压缩: log(1 + α·I), α=100
    5. 再归一化到 [0, 1]

运行:
    python generate_data.py

输出:
    data/
    ├── apertures.npy           (N, 256, 256) 光阑
    ├── patterns.npy            (N, 256, 256) 衍射图样 (log 压缩)
    ├── preview_strategy.png    # 每种策略的样例
    └── preview_random.png      # 随机样本预览
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# ============================================================
# 配置 (必须和 train.py 一致)
# ============================================================
GRID = 256
N_SAMPLES = 5000
OUTPUT_DIR = 'data'
SEED = 42

# ⚠️ 关键参数: 必须和 train.py 里的 LOG_ALPHA 一致
LOG_ALPHA = 100.0

# 噪声强度
APERTURE_NOISE = 0.005

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 绘图工具
# ============================================================
def draw_rect(a, cx, cy, w, h, angle=0):
    """旋转矩形"""
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    mask = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
    a[mask] = 1.0


def draw_circle(a, cx, cy, r):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r**2
    a[mask] = 1.0


def draw_ellipse(a, cx, cy, rx, ry, angle=0):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    mask = (xr / rx)**2 + (yr / ry)**2 <= 1
    a[mask] = 1.0


def draw_ring(a, cx, cy, r_in, r_out):
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    d2 = (xx - cx)**2 + (yy - cy)**2
    mask = (d2 >= r_in**2) & (d2 <= r_out**2)
    a[mask] = 1.0


def draw_polygon(a, cx, cy, n_sides, radius, angle=0):
    """正多边形"""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle
    verts = np.stack([cx + radius * np.cos(angles),
                      cy + radius * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_irregular_polygon(a, cx, cy, n_sides, r_min, r_max, angle=0,
                            rng=None):
    """不规则多边形: 每个顶点距离中心随机"""
    if rng is None:
        rng = np.random
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle
    angles = angles + rng.uniform(-0.3, 0.3, n_sides)
    radii = rng.uniform(r_min, r_max, n_sides)
    verts = np.stack([cx + radii * np.cos(angles),
                      cy + radii * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_star(a, cx, cy, n_points, r_outer, r_inner, angle=0):
    """星形"""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False) + angle
    radii = np.tile([r_outer, r_inner], n_points)
    verts = np.stack([cx + radii * np.cos(angles),
                      cy + radii * np.sin(angles)], axis=1)
    _fill_polygon(a, verts)


def draw_cross(a, cx, cy, arm_length, arm_width, angle=0):
    """十字形"""
    draw_rect(a, cx, cy, arm_length, arm_width, angle)
    draw_rect(a, cx, cy, arm_width, arm_length, angle)


def _fill_polygon(a, verts):
    """射线法填充多边形"""
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    inside = np.zeros_like(xx, dtype=bool)
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cond = ((y1 > yy) != (y2 > yy))
        with np.errstate(divide='ignore', invalid='ignore'):
            x_int = (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-12) + x1
        inside ^= cond & (xx < x_int)
    a[inside] = 1.0


def draw_grating(a, cx, cy, w, h, period, duty, angle):
    """有限尺寸光栅"""
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    x = xx - cx
    y = yy - cy
    ca, sa = np.cos(-angle), np.sin(-angle)
    xr = ca * x - sa * y
    yr = sa * x + ca * y
    in_rect = (np.abs(xr) <= w / 2) & (np.abs(yr) <= h / 2)
    phase = (xr % period) / period
    bars = phase < duty
    a[in_rect & bars] = 1.0


def draw_multi_slits(a, cx, cy, n_slits, slit_width, gap, height, angle):
    """多缝"""
    total = n_slits * slit_width + (n_slits - 1) * gap
    start = -total / 2 + slit_width / 2
    for i in range(n_slits):
        offset = start + i * (slit_width + gap)
        dx = offset * np.cos(angle + np.pi / 2)
        dy = offset * np.sin(angle + np.pi / 2)
        draw_rect(a, cx + dx, cy + dy, slit_width, height, angle)


# ============================================================
# 8 种光阑生成策略
# ============================================================
def strategy_multi_elements(a, rng, size):
    """策略1: 多个不同元素随机放置"""
    n = rng.randint(2, 5)
    placed = []
    
    for _ in range(n):
        for _attempt in range(15):
            cx = rng.randint(40, size - 40)
            cy = rng.randint(40, size - 40)
            ok = True
            for (px, py, pr) in placed:
                if (cx - px)**2 + (cy - py)**2 < (pr + 30)**2:
                    ok = False
                    break
            if ok:
                break
        
        elem = rng.choice(['rect', 'circle', 'ellipse', 'polygon'])
        if elem == 'rect':
            w, h = rng.randint(15, 60), rng.randint(15, 60)
            ang = rng.uniform(0, np.pi)
            draw_rect(a, cx, cy, w, h, ang)
            placed.append((cx, cy, max(w, h)))
        elif elem == 'circle':
            r = rng.randint(10, 35)
            draw_circle(a, cx, cy, r)
            placed.append((cx, cy, r))
        elif elem == 'ellipse':
            rx, ry = rng.randint(10, 45), rng.randint(10, 45)
            ang = rng.uniform(0, np.pi)
            draw_ellipse(a, cx, cy, rx, ry, ang)
            placed.append((cx, cy, max(rx, ry)))
        else:
            n_sides = rng.randint(3, 8)
            r = rng.randint(15, 40)
            ang = rng.uniform(0, 2 * np.pi)
            draw_polygon(a, cx, cy, n_sides, r, ang)
            placed.append((cx, cy, r))


def strategy_slits(a, rng, size):
    """策略2: 多缝"""
    n = rng.choice([1, 2, 3, 4, 5])
    slit_width = rng.randint(3, 12)
    gap = rng.randint(15, 60)
    height = rng.randint(80, 180)
    angle = rng.uniform(0, np.pi)
    cx = size / 2 + rng.randint(-30, 30)
    cy = size / 2 + rng.randint(-30, 30)
    draw_multi_slits(a, cx, cy, n, slit_width, gap, height, angle)


def strategy_grating(a, rng, size):
    """策略3: 有限尺寸光栅"""
    w = rng.randint(80, 200)
    h = rng.randint(80, 200)
    period = rng.randint(8, 30)
    duty = rng.uniform(0.3, 0.7)
    angle = rng.uniform(0, np.pi)
    cx = size / 2 + rng.randint(-40, 40)
    cy = size / 2 + rng.randint(-40, 40)
    draw_grating(a, cx, cy, w, h, period, duty, angle)


def strategy_hole_array(a, rng, size):
    """策略4: 孔洞阵列"""
    pattern = rng.choice(['grid', 'ring', 'random'])
    
    if pattern == 'grid':
        n_x = rng.randint(3, 8)
        n_y = rng.randint(3, 8)
        r = rng.randint(3, 10)
        margin = 40
        xs = np.linspace(margin, size - margin, n_x)
        ys = np.linspace(margin, size - margin, n_y)
        for x in xs:
            for y in ys:
                draw_circle(a, x, y, r)
    
    elif pattern == 'ring':
        n = rng.randint(6, 16)
        r_hole = rng.randint(3, 8)
        r_ring = rng.randint(40, 90)
        cx, cy = size / 2, size / 2
        for k in range(n):
            ang = 2 * np.pi * k / n + rng.uniform(-0.1, 0.1)
            x = cx + r_ring * np.cos(ang)
            y = cy + r_ring * np.sin(ang)
            draw_circle(a, x, y, r_hole)
    
    else:
        n = rng.randint(5, 20)
        for _ in range(n):
            cx = rng.randint(30, size - 30)
            cy = rng.randint(30, size - 30)
            r = rng.randint(3, 12)
            draw_circle(a, cx, cy, r)


def strategy_letter(a, rng, size):
    """策略5: 字母形状"""
    letter = rng.choice(['T', 'L', 'H', 'X', 'O', 'C', 'F', 'E'])
    thickness = rng.randint(8, 18)
    scale = rng.randint(60, 100)
    cx, cy = size / 2, size / 2
    
    if letter == 'T':
        draw_rect(a, cx, cy - scale / 2, scale, thickness, 0)
        draw_rect(a, cx, cy, thickness, scale, 0)
    elif letter == 'L':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx, cy + scale / 2, scale, thickness, 0)
    elif letter == 'H':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx + scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx, cy, scale, thickness, 0)
    elif letter == 'X':
        draw_rect(a, cx, cy, thickness, scale, np.pi / 4)
        draw_rect(a, cx, cy, thickness, scale, -np.pi / 4)
    elif letter == 'O':
        draw_ring(a, cx, cy, scale / 2 - thickness, scale / 2)
    elif letter == 'C':
        yy, xx = np.mgrid[:size, :size]
        d2 = (xx - cx)**2 + (yy - cy)**2
        outer = d2 <= (scale / 2)**2
        inner = d2 <= (scale / 2 - thickness)**2
        angle = np.arctan2(yy - cy, xx - cx)
        gap = angle > -np.pi / 4
        a[outer & ~inner & ~gap] = 1.0
    elif letter == 'F':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx - scale / 4, cy - scale / 2, scale / 2, thickness, 0)
        draw_rect(a, cx - scale / 4, cy, scale / 3, thickness, 0)
    elif letter == 'E':
        draw_rect(a, cx - scale / 2, cy, thickness, scale, 0)
        draw_rect(a, cx - scale / 4, cy - scale / 2, scale / 2, thickness, 0)
        draw_rect(a, cx - scale / 4, cy, scale / 3, thickness, 0)
        draw_rect(a, cx - scale / 4, cy + scale / 2, scale / 2, thickness, 0)


def strategy_annular(a, rng, size):
    """策略6: 环形/扇形"""
    pattern = rng.choice(['single_ring', 'multi_ring', 'sector', 'annular_array'])
    
    if pattern == 'single_ring':
        cx = size / 2 + rng.randint(-30, 30)
        cy = size / 2 + rng.randint(-30, 30)
        r_out = rng.randint(30, 80)
        r_in = r_out - rng.randint(5, 20)
        draw_ring(a, cx, cy, r_in, r_out)
    
    elif pattern == 'multi_ring':
        cx, cy = size / 2, size / 2
        n = rng.randint(2, 5)
        for k in range(n):
            r_out = rng.randint(20, 100)
            r_in = r_out - rng.randint(3, 8)
            draw_ring(a, cx, cy, r_in, r_out)
    
    elif pattern == 'sector':
        yy, xx = np.mgrid[:size, :size]
        cx, cy = size / 2, size / 2
        d2 = (xx - cx)**2 + (yy - cy)**2
        angle = np.arctan2(yy - cy, xx - cx)
        r_max = rng.randint(40, 100)
        n_sectors = rng.choice([2, 3, 4])
        sector_width = 2 * np.pi / n_sectors
        phase = rng.uniform(0, 2 * np.pi)
        for k in range(n_sectors):
            ang_center = phase + k * sector_width
            ang_mask = np.abs(np.angle(np.exp(1j * (angle - ang_center)))) < sector_width / 2.5
            a[(d2 <= r_max**2) & ang_mask] = 1.0
    
    else:
        cx, cy = size / 2, size / 2
        n_rings = rng.randint(2, 4)
        for k in range(1, n_rings + 1):
            r_out = 30 * k
            r_in = r_out - rng.randint(3, 8)
            if r_out < size // 2 - 10:
                draw_ring(a, cx, cy, r_in, r_out)


def strategy_mixed_grating(a, rng, size):
    """策略7: 光栅 + 缺陷"""
    w = rng.randint(100, 220)
    h = rng.randint(100, 220)
    period = rng.randint(8, 25)
    duty = rng.uniform(0.3, 0.7)
    angle = rng.uniform(0, np.pi)
    cx = size / 2 + rng.randint(-20, 20)
    cy = size / 2 + rng.randint(-20, 20)
    
    draw_grating(a, cx, cy, w, h, period, duty, angle)
    
    n_holes = rng.randint(1, 5)
    for _ in range(n_holes):
        hx = rng.randint(0, size)
        hy = rng.randint(0, size)
        hr = rng.randint(3, 15)
        yy, xx = np.mgrid[:size, :size]
        a[(xx - hx)**2 + (yy - hy)**2 <= hr**2] = 0.0
    
    if rng.rand() > 0.5:
        cx2 = rng.randint(30, size - 30)
        cy2 = rng.randint(30, size - 30)
        r2 = rng.randint(15, 40)
        draw_circle(a, cx2, cy2, r2)


def strategy_polygons(a, rng, size):
    """策略8: 各种多边形"""
    kind = rng.choice(['regular', 'irregular', 'star', 'cross', 'triangle_mix'])
    cx = size / 2 + rng.randint(-40, 40)
    cy = size / 2 + rng.randint(-40, 40)
    
    if kind == 'regular':
        n = rng.randint(3, 9)
        r = rng.randint(40, 90)
        ang = rng.uniform(0, 2 * np.pi)
        draw_polygon(a, cx, cy, n, r, ang)
    
    elif kind == 'irregular':
        n = rng.randint(3, 8)
        r_min, r_max = 30, 90
        ang = rng.uniform(0, 2 * np.pi)
        draw_irregular_polygon(a, cx, cy, n, r_min, r_max, ang, rng)
    
    elif kind == 'star':
        n = rng.randint(3, 8)
        r_out = rng.randint(50, 100)
        r_in = r_out * rng.uniform(0.3, 0.6)
        ang = rng.uniform(0, 2 * np.pi)
        draw_star(a, cx, cy, n, r_out, r_in, ang)
    
    elif kind == 'cross':
        arm = rng.randint(60, 120)
        width = rng.randint(8, 25)
        ang = rng.uniform(0, np.pi)
        draw_cross(a, cx, cy, arm, width, ang)
    
    else:
        n = rng.randint(2, 5)
        for _ in range(n):
            tcx = rng.randint(40, size - 40)
            tcy = rng.randint(40, size - 40)
            r = rng.randint(20, 50)
            ang = rng.uniform(0, 2 * np.pi)
            draw_polygon(a, tcx, tcy, 3, r, ang)


# ============================================================
# 随机光阑
# ============================================================
def random_aperture(size=256, rng=None):
    """随机光阑，8 种策略随机选一"""
    if rng is None:
        rng = np.random
    
    a = np.zeros((size, size), dtype=np.float32)
    
    strategy = rng.choice([
        'multi_elements',
        'slits',
        'grating',
        'hole_array',
        'letter',
        'annular',
        'mixed_grating',
        'polygons',
    ])
    
    if strategy == 'multi_elements':
        strategy_multi_elements(a, rng, size)
    elif strategy == 'slits':
        strategy_slits(a, rng, size)
    elif strategy == 'grating':
        strategy_grating(a, rng, size)
    elif strategy == 'hole_array':
        strategy_hole_array(a, rng, size)
    elif strategy == 'letter':
        strategy_letter(a, rng, size)
    elif strategy == 'annular':
        strategy_annular(a, rng, size)
    elif strategy == 'mixed_grating':
        strategy_mixed_grating(a, rng, size)
    elif strategy == 'polygons':
        strategy_polygons(a, rng, size)
    
    # 轻微透过率变化
    if rng.rand() > 0.5:
        smooth = rng.uniform(0.85, 1.0)
        a = a * smooth
    
    # 轻微噪声
    a = a + rng.randn(size, size) * APERTURE_NOISE
    a = np.clip(a, 0, 1)
    
    return a


# ============================================================
# 物理模拟 (必须和 train.py 的 FraunhoferLayer 完全一致)
# ============================================================
def simulate_diffraction(aperture):
    """
    远场夫琅禾费衍射。
    
    处理流程:
        1. FFT + fftshift
        2. |.|²
        3. 归一化
        4. log 压缩
        5. 再归一化
    
    与 train.py 的 FraunhoferLayer.forward 完全一致。
    """
    # 复数
    field = aperture.astype(np.complex64)
    
    # FFT + fftshift
    field_f = np.fft.fft2(field)
    field_f = np.fft.fftshift(field_f)
    
    # 强度
    intensity = np.abs(field_f) ** 2
    
    # 第 1 次归一化
    intensity = intensity / (intensity.max() + 1e-12)
    
    # log 压缩
    intensity = np.log1p(LOG_ALPHA * intensity)
    
    # 第 2 次归一化
    intensity = intensity / (intensity.max() + 1e-12)
    
    return intensity.astype(np.float32)


# ============================================================
# 主流程
# ============================================================
def main():
    print("=" * 60)
    print("生成模拟数据")
    print("=" * 60)
    print(f"样本数: {N_SAMPLES}")
    print(f"图像尺寸: {GRID}×{GRID}")
    print(f"LOG_ALPHA: {LOG_ALPHA}")
    print(f"光阑噪声: {APERTURE_NOISE}")
    print()
    
    # 一致性检查: 手动模拟一个圆孔
    print("一致性检查: 用圆孔测试物理层")
    yy, xx = np.ogrid[:GRID, :GRID]
    test_a = ((xx - GRID//2)**2 + (yy - GRID//2)**2 <= 30**2).astype(np.float32)
    test_p = simulate_diffraction(test_a)
    print(f"  输入: 圆孔 (r=30)")
    print(f"  输出: [{test_p.min():.4f}, {test_p.max():.4f}]  mean={test_p.mean():.4f}")
    print(f"  期望: 中心亮斑 + 同心圆环 (Airy 斑)")
    print()
    
    # 生成数据
    apertures = np.zeros((N_SAMPLES, GRID, GRID), dtype=np.float32)
    patterns = np.zeros((N_SAMPLES, GRID, GRID), dtype=np.float32)
    
    rng = np.random.RandomState(SEED)
    
    print("生成数据...")
    for i in range(N_SAMPLES):
        a = random_aperture(GRID, rng)
        p = simulate_diffraction(a)
        apertures[i] = a
        patterns[i] = p
        
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i+1}/{N_SAMPLES}")
    
    # 保存
    np.save(os.path.join(OUTPUT_DIR, 'apertures.npy'), apertures)
    np.save(os.path.join(OUTPUT_DIR, 'patterns.npy'), patterns)
    print(f"\n保存到 {OUTPUT_DIR}/")
    print(f"  apertures.npy: {apertures.shape}")
    print(f"  patterns.npy:  {patterns.shape}")
    
    # 数据统计
    print("\n数据统计:")
    print(f"  光阑:  [{apertures.min():.4f}, {apertures.max():.4f}]  "
          f"mean={apertures.mean():.4f}")
    print(f"  衍射:  [{patterns.min():.4f}, {patterns.max():.4f}]  "
          f"mean={patterns.mean():.4f}")
    
    # ============ 预览 1: 每种策略 ============
    print("\n生成预览图...")
    strategies = {
        'multi_elements': strategy_multi_elements,
        'slits': strategy_slits,
        'grating': strategy_grating,
        'hole_array': strategy_hole_array,
        'letter': strategy_letter,
        'annular': strategy_annular,
        'mixed_grating': strategy_mixed_grating,
        'polygons': strategy_polygons,
    }
    
    n_strat = len(strategies)
    fig, axes = plt.subplots(n_strat, 4, figsize=(16, 3 * n_strat))
    
    rng2 = np.random.RandomState(123)
    for row, (name, fn) in enumerate(strategies.items()):
        for col in range(2):
            a = np.zeros((GRID, GRID), dtype=np.float32)
            fn(a, rng2, GRID)
            p = simulate_diffraction(a)
            
            axes[row][col*2].imshow(a, cmap='gray', vmin=0, vmax=1)
            axes[row][col*2].set_title(f'{name}', fontsize=9)
            axes[row][col*2].axis('off')
            
            axes[row][col*2+1].imshow(p, cmap='inferno', vmin=0, vmax=1)
            axes[row][col*2+1].set_title(f'{name} - diffraction', fontsize=9)
            axes[row][col*2+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'preview_strategy.png'), dpi=90)
    plt.close()
    print(f"  → {OUTPUT_DIR}/preview_strategy.png")
    
    # ============ 预览 2: 随机样本 ============
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    for i in range(5):
        for j in range(3):
            idx = rng.randint(0, N_SAMPLES)
            axes[i][j*2].imshow(apertures[idx], cmap='gray', vmin=0, vmax=1)
            axes[i][j*2].set_title(f'#{idx} aperture', fontsize=8)
            axes[i][j*2].axis('off')
            
            axes[i][j*2+1].imshow(patterns[idx], cmap='inferno', vmin=0, vmax=1)
            axes[i][j*2+1].set_title(f'#{idx} diffraction', fontsize=8)
            axes[i][j*2+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'preview_random.png'), dpi=90)
    plt.close()
    print(f"  → {OUTPUT_DIR}/preview_random.png")
    
    # ============ 预览 3: 中心行剖面 ============
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    
    # 单缝
    a = np.zeros((GRID, GRID), dtype=np.float32)
    a[40:GRID-40, GRID//2-4:GRID//2+4] = 1.0
    p = simulate_diffraction(a)
    axes[0].imshow(p, cmap='inferno', aspect='auto')
    axes[0].set_title('Single slit - diffraction')
    
    # 圆孔
    a = ((xx - GRID//2)**2 + (yy - GRID//2)**2 <= 30**2).astype(np.float32)
    p = simulate_diffraction(a)
    axes[1].imshow(p, cmap='inferno', aspect='auto')
    axes[1].set_title('Circle - Airy pattern')
    
    # 光栅
    a = np.zeros((GRID, GRID), dtype=np.float32)
    for x in range(0, GRID, 16):
        a[:, x:x+6] = 1.0
    p = simulate_diffraction(a)
    axes[2].imshow(p, cmap='inferno', aspect='auto')
    axes[2].set_title('Grating - discrete orders')
    
    # 随机样本
    p = patterns[rng.randint(0, N_SAMPLES)]
    axes[3].imshow(p, cmap='inferno', aspect='auto')
    axes[3].set_title('Random sample')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'preview_sanity.png'), dpi=90)
    plt.close()
    print(f"  → {OUTPUT_DIR}/preview_sanity.png")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  1. 打开 data/preview_sanity.png 确认物理正确")
    print("  2. 打开 data/preview_strategy.png 看多样性")
    print("  3. 运行 python train.py 训练")


if __name__ == '__main__':
    main()