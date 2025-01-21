#@+leo-ver=5-thin
#@+node:ekr.20250120131606.1: * @file example.py
"""Example file from community addition of manim"""

from manim import *

class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        square.flip(RIGHT)
        square.rotate(-3 * TAU / 8)
        circle.set_fill(PINK, opacity=0.5)

        self.play(Create(square))
        self.play(Transform(square, circle))
        self.play(FadeOut(square))
#@-leo
