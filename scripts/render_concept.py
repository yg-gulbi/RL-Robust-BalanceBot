"""Render an illustrative goal video, NOT an RL rollout or physics validation.

The approximate robot is inspired by the supplied screenshot. All poses are
scripted. MuJoCo is used only for 3D rendering; mj_step is deliberately not used.
"""

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT, FPS, CHAPTER_SECONDS = 960, 600, 24, 6
ACCENT = (54, 215, 179)
FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CHAPTERS = [
    ("01", "외부에서 밀려도", "흔들린 자세를 회복하고, 다시 이동하기", "외란 · 자세 회복"),
    ("02", "경사면에서도", "기울어진 바닥에서 균형을 잡으며 이동하기", "경사면 · 균형 유지"),
    ("03", "고르지 않은 바닥에서도", "바닥 변화에 대응하며 원하는 방향으로 이동하기", "노면 변화 · 이동 제어"),
]


def ground_height(x, chapter):
    if chapter == 1:
        return math.tan(math.radians(9)) * x
    if chapter == 2:
        return 0.025 * math.sin(5 * x) + 0.012 * math.sin(11 * x)
    return 0.0


def scene_xml(chapter):
    terrain = ""
    for index, x in enumerate(np.arange(-3.5, 3.5, 0.12)):
        z = ground_height(float(x), chapter)
        shade = "0.29 0.35 0.41 1" if index % 2 else "0.32 0.38 0.44 1"
        terrain += f'<geom type="box" pos="{x} 0 {z - .10}" size=".061 1.3 .10" rgba="{shade}"/>'
        for y in (-0.78, 0.78):
            terrain += f'<geom type="box" pos="{x} {y} {z + .002}" size=".058 .012 .003" rgba=".23 .84 .70 1"/>'
    posts = ""
    for x in (-2.8, -1.4, 0, 1.4, 2.8):
        for y in (-1.12, 1.12):
            z = ground_height(x, chapter)
            posts += f'<geom type="box" pos="{x} {y} {z + .09}" size=".028 .028 .09" rgba=".48 .56 .61 1"/>'
    return f'''<mujoco model="balancebot_goal_illustration">
      <compiler angle="degree"/>
      <visual>
        <global offwidth="{WIDTH}" offheight="{HEIGHT}"/>
        <quality shadowsize="2048"/>
        <headlight ambient=".42 .42 .42" diffuse=".55 .55 .55" specular=".2 .2 .2"/>
        <rgba haze=".10 .13 .17 1"/>
      </visual>
      <asset>
        <texture name="sky" type="skybox" builtin="gradient" rgb1=".09 .12 .17" rgb2=".27 .33 .39" width="512" height="512"/>
        <texture name="grid" type="2d" builtin="checker" rgb1=".18 .23 .29" rgb2=".20 .25 .31" width="512" height="512"/>
        <material name="ground" texture="grid" texrepeat="28 28" texuniform="true" reflectance=".06"/>
        <material name="shell" rgba=".065 .078 .088 1" specular=".65" shininess=".55"/>
        <material name="metal" rgba=".12 .15 .17 1" specular=".6" shininess=".6"/>
        <material name="rubber" rgba=".032 .037 .043 1" specular=".08"/>
      </asset>
      <worldbody>
        <light pos="-2 -3 6" dir=".2 .3 -1" diffuse=".9 .92 .95" castshadow="true"/>
        <light pos="3 2 4" dir="-.4 -.3 -1" diffuse=".6 .75 .82" castshadow="false"/>
        <geom type="plane" pos="0 0 -1" size="30 30 .1" material="ground"/>
        {terrain}{posts}
        <body name="robot" mocap="true" pos="0 0 .18">
          <geom type="box" pos="0 0 .14" size=".24 .19 .12" material="shell"/>
          <geom type="box" pos="0 0 .295" size=".20 .175 .035" material="metal"/>
          <geom type="box" pos=".247 0 .16" size=".005 .12 .026" rgba=".22 .85 .71 1"/>
          <geom type="box" pos="-.12 -.198 .16" size=".065 .003 .022" rgba=".3 .34 .38 1"/>
          <geom type="box" pos="-.12 .198 .16" size=".065 .003 .022" rgba=".3 .34 .38 1"/>
          <geom type="cylinder" pos="0 0 .39" size=".048 .045" material="metal"/>
          <geom type="box" pos="0 0 .46" size=".135 .13 .025" material="shell"/>
          <geom type="box" pos="0 0 .69" size=".135 .13 .025" material="shell"/>
          <geom type="box" pos="-.112 -.107 .575" size=".014 .014 .10" material="metal"/>
          <geom type="box" pos=".112 -.107 .575" size=".014 .014 .10" material="metal"/>
          <geom type="box" pos="-.112 .107 .575" size=".014 .014 .10" material="metal"/>
          <geom type="box" pos=".112 .107 .575" size=".014 .014 .10" material="metal"/>
          <geom type="cylinder" pos="0 0 .733" size=".066 .025" material="shell"/>
          <geom type="cylinder" pos="0 0 .76" size=".057 .006" rgba=".23 .84 .70 1"/>
          <body name="left_wheel" pos="0 .255 0">
            <joint type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size=".18 .065" euler="90 0 0" material="rubber"/>
            <geom type="cylinder" pos="0 .068 0" size=".105 .004" euler="90 0 0" material="metal"/>
            <geom type="box" pos="0 .075 0" size=".09 .004 .012" rgba=".43 .49 .54 1"/>
            <geom type="box" pos="0 .075 0" size=".012 .004 .09" rgba=".43 .49 .54 1"/>
          </body>
          <body name="right_wheel" pos="0 -.255 0">
            <joint type="hinge" axis="0 1 0"/>
            <geom type="cylinder" size=".18 .065" euler="90 0 0" material="rubber"/>
            <geom type="cylinder" pos="0 -.068 0" size=".105 .004" euler="90 0 0" material="metal"/>
            <geom type="box" pos="0 -.075 0" size=".09 .004 .012" rgba=".43 .49 .54 1"/>
            <geom type="box" pos="0 -.075 0" size=".012 .004 .09" rgba=".43 .49 .54 1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>'''


def scripted_pose(t, chapter):
    # Kinematic choreography only: these values are not controller outputs.
    x = -0.9 + 1.8 * t / CHAPTER_SECONDS
    pitch = 0.018 * math.sin(2 * t)
    phase = t - 1.5
    if chapter == 0 and phase >= 0:
        pitch += 0.21 * math.exp(-1.45 * phase) * math.sin(7 * phase)
        x += 0.11 * math.exp(-1.1 * phase) * math.sin(5 * phase)
    elif chapter == 1:
        pitch += 0.035
    elif chapter == 2:
        pitch += 0.055 * math.sin(8 * t)
    return x, ground_height(x, chapter) + 0.18, pitch


def draw_overlay(rgb, chapter, t, fonts):
    frame = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, WIDTH, 115), fill=(12, 20, 31, 244))
    draw.text((30, 15), "RL-Robust-BalanceBot", font=fonts[21], fill=ACCENT)
    draw.text((30, 46), CHAPTERS[chapter][1], font=fonts[31], fill=(244, 248, 252))
    draw.text((31, 88), CHAPTERS[chapter][2], font=fonts[16], fill=(177, 192, 206))
    draw.rounded_rectangle((775, 20, 930, 53), radius=16, fill=(31, 49, 61))
    draw.text((795, 25), "목표 동작 예시", font=fonts[16], fill=ACCENT)
    draw.rectangle((0, 485, WIDTH, HEIGHT), fill=(12, 20, 31, 246))
    for i, (_, _, _, label) in enumerate(CHAPTERS):
        x0 = 30 + i * 308
        color = ACCENT if i == chapter else (75, 91, 109)
        draw.rounded_rectangle((x0, 508, x0 + 284, 513), radius=2, fill=(37, 53, 69))
        if i < chapter:
            draw.rounded_rectangle((x0, 508, x0 + 284, 513), radius=2, fill=ACCENT)
        elif i == chapter:
            draw.rounded_rectangle((x0, 508, x0 + max(4, int(284 * t / CHAPTER_SECONDS)), 513), radius=2, fill=ACCENT)
        draw.text((x0, 522), label, font=fonts[18], fill=color)
    draw.text((30, 568), "컨셉 영상 · 동작 연출 · 실제 로봇 / 강화학습 결과 아님", font=fonts[15], fill=(177, 192, 206))
    if chapter == 0 and 1.5 <= t < 2.25:
        draw.rounded_rectangle((660, 163, 895, 202), radius=10, fill=(163, 84, 27, 235))
        draw.text((677, 170), "외부에서 밀리는 상황", font=fonts[18], fill=(255, 236, 200))
    return Image.alpha_composite(frame, overlay).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/media")
    parser.add_argument("--font", type=Path, default=FONT_PATH,
                        help="Path to a Korean font, e.g. Noto Sans CJK")
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("A Korean font is required. Install Noto Sans CJK or pass --font.")
    args.output.mkdir(parents=True, exist_ok=True)
    fonts = {size: ImageFont.truetype(str(args.font), size, index=1)
             for size in (15, 16, 18, 21, 31)}
    video_path = args.output / "balancebot-concept.mp4"
    writer = imageio_ffmpeg.write_frames(
        str(video_path), (WIDTH, HEIGHT), fps=FPS, codec="libx264",
        pix_fmt_out="yuv420p", quality=8, macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    gif_frames = []
    try:
        for chapter in range(len(CHAPTERS)):
            model = mujoco.MjModel.from_xml_string(scene_xml(chapter))
            data = mujoco.MjData(model)
            camera = mujoco.MjvCamera()
            camera.azimuth, camera.elevation, camera.distance = 135, -18, 3.5
            camera.lookat[:] = (0, 0, .43)
            with mujoco.Renderer(model, HEIGHT, WIDTH) as renderer:
                for index in range(FPS * CHAPTER_SECONDS):
                    t = index / FPS
                    x, z, pitch = scripted_pose(t, chapter)
                    data.mocap_pos[0] = (x, 0, z)
                    data.mocap_quat[0] = (math.cos(pitch / 2), 0, math.sin(pitch / 2), 0)
                    data.qpos[:] = x / .18
                    mujoco.mj_forward(model, data)
                    camera.lookat[0] = x * .45
                    renderer.update_scene(data, camera=camera)
                    if chapter == 0 and 1.5 <= t < 1.85:
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW,
                                           np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                           np.array([1, .48, .18, 1], dtype=np.float32))
                        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, .025,
                                            [x - .85, 0, z + .38], [x - .22, 0, z + .38])
                        renderer.scene.ngeom += 1
                    frame = draw_overlay(renderer.render(), chapter, t, fonts)
                    writer.send(np.asarray(frame))
                    if index % 3 == 0:
                        gif_frames.append(frame.resize((640, 400), Image.Resampling.LANCZOS))
            print(f"Rendered chapter {chapter + 1}/{len(CHAPTERS)}", flush=True)
    finally:
        writer.close()
    # Use one shared palette to avoid flicker between frames.
    montage = Image.new("RGB", (640, 400 * len(CHAPTERS)))
    for i in range(len(CHAPTERS)):
        montage.paste(gif_frames[i * FPS * CHAPTER_SECONDS // 3 + 24], (0, 400 * i))
    palette = montage.quantize(colors=96)
    indexed = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in gif_frames]
    indexed[0].save(args.output / "balancebot-concept.gif", save_all=True,
                    append_images=indexed[1:], duration=[120, 130] * (len(indexed) // 2),
                    loop=0, optimize=True, disposal=1)
    print(f"Saved {video_path} and GIF ({len(gif_frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
