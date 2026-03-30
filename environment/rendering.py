"""
rendering.py — Pygame 2D visualization for the Eco-Scale Kubernetes cluster.

Shows a node grid with load-based coloring, real-time metrics bars,
and action/reward log. Can be run standalone for a random-agent demo.
"""

import sys
import os
import numpy as np

# ---------- Color palette ----------
COLORS = {
    "bg": (18, 18, 24),
    "panel": (30, 30, 42),
    "panel_border": (55, 55, 75),
    "text": (220, 220, 230),
    "text_dim": (140, 140, 160),
    "title_bg": (40, 40, 58),
    "green": (29, 158, 117),       # #1D9E75  — idle / low load
    "amber": (239, 159, 39),       # #EF9F27  — normal load
    "red": (226, 75, 74),          # #E24B4A  — high load
    "gray": (180, 178, 169),       # #B4B2A9  — inactive node
    "bar_bg": (50, 50, 65),
    "accent": (100, 140, 255),
    "reward_pos": (80, 200, 120),
    "reward_neg": (226, 75, 74),
}

# Node colors by CPU utilization
def _node_color(cpu, active):
    if not active:
        return COLORS["gray"]
    if cpu < 0.3:
        return COLORS["green"]
    elif cpu < 0.7:
        return COLORS["amber"]
    else:
        return COLORS["red"]


# ---------- Action labels ----------
ACTION_LABELS = {0: "SCALE DOWN", 1: "HOLD", 2: "SCALE UP"}


def init_pygame(width=860, height=540):
    """Initialize pygame window and return surface."""
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ECO-SCALE — Kubernetes Cluster Monitor")
    return screen


def _draw_rounded_rect(surface, color, rect, radius=8):
    """Draw a rounded rectangle."""
    import pygame
    x, y, w, h = rect
    # Clamp radius
    radius = min(radius, w // 2, h // 2)
    pygame.draw.rect(surface, color, rect, border_radius=radius)


def _draw_bar(surface, x, y, width, height, fraction, color, bg_color=None):
    """Draw a horizontal bar chart."""
    import pygame
    bg = bg_color or COLORS["bar_bg"]
    pygame.draw.rect(surface, bg, (x, y, width, height), border_radius=4)
    fill_w = max(0, min(int(width * fraction), width))
    if fill_w > 0:
        pygame.draw.rect(surface, color, (x, y, fill_w, height), border_radius=4)


def draw_cluster(screen, pod_count, cpu_util, font):
    """Draw the 10-node grid with load-based coloring."""
    import pygame
    start_x, start_y = 30, 80
    node_size = 60
    gap = 10
    cols = 5

    for i in range(10):
        row, col = divmod(i, cols)
        x = start_x + col * (node_size + gap)
        y = start_y + row * (node_size + gap)
        active = i < pod_count
        color = _node_color(cpu_util, active)
        _draw_rounded_rect(screen, color, (x, y, node_size, node_size), radius=6)
        # Node label
        label = font.render(f"P{i+1}", True, COLORS["bg"] if active else COLORS["text_dim"])
        lx = x + (node_size - label.get_width()) // 2
        ly = y + (node_size - label.get_height()) // 2
        screen.blit(label, (lx, ly))


def draw_metrics(screen, env_state, font, font_small):
    """Draw bar charts for CPU, pods, queue, latency, and status text."""
    import pygame
    bar_x = 400
    bar_w = 320
    bar_h = 22
    y_start = 90
    gap = 50

    cpu = env_state.get("cpu_util", 0)
    pods = env_state.get("pods", 0)
    queue = env_state.get("request_queue", 0)
    latency = env_state.get("latency", 0)
    step = env_state.get("step", 0)
    tod = env_state.get("time_of_day", 0)

    metrics = [
        ("CPU Load", cpu, COLORS["amber"] if cpu < 0.7 else COLORS["red"], f"{cpu*100:.0f}%"),
        ("Pods", pods / 20.0, COLORS["accent"], f"{pods} / 20"),
        ("Queue", min(queue / 1000.0, 1.0), COLORS["amber"], f"{queue} reqs"),
        ("Latency", min(latency, 1.0), COLORS["red"] if latency > 0.5 else COLORS["green"], f"{latency:.2f}"),
    ]

    for i, (label, frac, color, value_str) in enumerate(metrics):
        y = y_start + i * gap
        lbl = font.render(label, True, COLORS["text_dim"])
        screen.blit(lbl, (bar_x, y - 20))
        _draw_bar(screen, bar_x, y, bar_w, bar_h, frac, color)
        val = font_small.render(value_str, True, COLORS["text"])
        screen.blit(val, (bar_x + bar_w + 10, y + 1))

    # Time / step info
    y_info = y_start + len(metrics) * gap + 10
    time_str = f"Time: {tod:02d}:00    Step: {step} / 288"
    info_surf = font.render(time_str, True, COLORS["text_dim"])
    screen.blit(info_surf, (bar_x, y_info))


def draw_action_log(screen, last_action, last_reward, total_reward, font, font_big):
    """Display last action taken and reward received."""
    y_base = 380

    # Action label
    action_str = ACTION_LABELS.get(last_action, "—")
    act_surf = font_big.render(f"Action: {action_str}", True, COLORS["text"])
    screen.blit(act_surf, (30, y_base))

    # Last reward
    r_color = COLORS["reward_pos"] if last_reward >= 0 else COLORS["reward_neg"]
    r_surf = font.render(f"Step Reward: {last_reward:+.3f}", True, r_color)
    screen.blit(r_surf, (30, y_base + 40))

    # Episode reward
    er_color = COLORS["reward_pos"] if total_reward >= 0 else COLORS["reward_neg"]
    er_surf = font.render(f"Episode Reward: {total_reward:+.2f}", True, er_color)
    screen.blit(er_surf, (30, y_base + 70))


def render_frame(screen, env_state, last_action, last_reward, total_reward):
    """Master render call — called once per step."""
    import pygame

    # Lazy-init fonts
    if not hasattr(render_frame, "_fonts"):
        render_frame._fonts = {
            "small": pygame.font.SysFont("monospace", 14),
            "normal": pygame.font.SysFont("monospace", 16),
            "big": pygame.font.SysFont("monospace", 20, bold=True),
            "title": pygame.font.SysFont("monospace", 22, bold=True),
        }
    fonts = render_frame._fonts

    # Background
    screen.fill(COLORS["bg"])

    # Title bar
    _draw_rounded_rect(screen, COLORS["title_bg"], (0, 0, 860, 55), radius=0)
    title = fonts["title"].render("ECO-SCALE  —  Kubernetes Cluster Monitor", True, COLORS["text"])
    screen.blit(title, (20, 15))

    # Cluster grid
    pod_count = env_state.get("pods", 0)
    cpu_util = env_state.get("cpu_util", 0)
    draw_cluster(screen, pod_count, cpu_util, fonts["normal"])

    # Metric bars
    draw_metrics(screen, env_state, fonts["normal"], fonts["small"])

    # Action log
    draw_action_log(screen, last_action, last_reward, total_reward, fonts["normal"], fonts["big"])

    # Legend
    y_leg = 490
    for label, color in [("Low", COLORS["green"]), ("Normal", COLORS["amber"]),
                          ("High", COLORS["red"]), ("Inactive", COLORS["gray"])]:
        pygame.draw.rect(screen, color, (30, y_leg, 14, 14), border_radius=3)
        leg_surf = fonts["small"].render(label, True, COLORS["text_dim"])
        screen.blit(leg_surf, (50, y_leg))
        y_leg += 0
        # shift horizontally instead
    # Redo legend horizontally
    screen.fill(COLORS["bg"], (30, 490, 830, 40))
    x_leg = 30
    for label, color in [("Low <30%", COLORS["green"]), ("Normal 30-70%", COLORS["amber"]),
                          ("High >70%", COLORS["red"]), ("Inactive", COLORS["gray"])]:
        pygame.draw.rect(screen, color, (x_leg, 500, 14, 14), border_radius=3)
        leg_surf = fonts["small"].render(label, True, COLORS["text_dim"])
        screen.blit(leg_surf, (x_leg + 20, 500))
        x_leg += len(label) * 9 + 45

    pygame.display.flip()


# ---------- Standalone random-agent demo ----------
def _run_random_demo():
    """Run the environment with random actions to demo visualization."""
    import pygame
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from environment.custom_env import KubernetesEnv

    env = KubernetesEnv(trace_type="cyclical")
    screen = init_pygame()
    clock = pygame.time.Clock()

    obs, _ = env.reset()
    total_reward = 0.0
    last_action = 1
    last_reward = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        last_action = action
        last_reward = reward

        render_frame(screen, info, last_action, last_reward, total_reward)
        clock.tick(4)

        if terminated or truncated:
            print(f"Episode complete. Total reward: {total_reward:.2f}")
            obs, _ = env.reset()
            total_reward = 0.0

    pygame.quit()


if __name__ == "__main__":
    _run_random_demo()
