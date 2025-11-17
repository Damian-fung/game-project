3D Snake (Enhanced)

Overview
--------
This project is a 3D snake-like game using Pygame + PyOpenGL. The repository includes a playable prototype with AI snakes, environment systems, and rendering code.

Quick goals implemented
- Runable demo using Pygame + PyOpenGL
- GLU spheres for better appearance
- Improved lighting and camera smoothing

Requirements
------------
Recommended Python: 3.10+ (you are using 3.13 in your logs; that should work but TensorFlow is optional).

Required packages (install using pip):
- pygame
- PyOpenGL
- numpy

Optional (heavy):
- tensorflow  # only needed if you plan to run/train agents; large install

Example installation

```bash
python -m pip install -r requirements.txt
```

Run
---

Usage
-----
To run the basic snake game (with visible window):
```bash
python game.py
```

To run AI training (no game window, only logs):
```bash
RUN_TRAIN=1 python game.py
```

Notes & Troubleshooting
-----------------------
- On Windows, avoid mixing GLUT window creation with Pygame; this project uses Pygame to create the OpenGL context and PyOpenGL GLU for spheres.
- If display fails, check your GPU drivers and ensure hardware acceleration for OpenGL is available.

Next steps (suggested)
- Replace immediate-mode drawing with modern OpenGL (VBO/VAO + GLSL shaders)
- Add textures and better materials
- Improve UI/HUD using Pygame overlays

If you want, I can proceed to convert rendering to shaders and VBOs (this is larger but yields the best game-quality visuals).# game-project
