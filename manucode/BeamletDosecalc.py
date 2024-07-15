import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


def draw_cube():
    # Cube vertices
    vertices = (
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1)
    )

    # Cube edges
    edges = (
        (0, 1), (0, 3), (0, 4), (2, 1), (2, 3), (2, 7),
        (6, 3), (6, 4), (6, 7), (5, 1), (5, 4), (5, 7)
    )

    # Cube faces
    faces = (
        (0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4),
        (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6)
    )

    # Face colors (RGBA)
    colors = (
        (1, 0, 0, 0.5), (0, 1, 0, 0.5), (0, 0, 1, 0.5),
        (1, 1, 0, 0.5), (1, 0, 1, 0.5), (0, 1, 1, 0.5)
    )
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor4fv(colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    glColor3fv((1, 1, 1))
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def cubeDemo():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    clock = pygame.time.Clock()
    
    rotation = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        glRotatef(1, 3, 1, 1)
        rotation += 1

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_cube()
        pygame.display.flip()
        
        if rotation == 90:  # Save image after 90 degrees of rotation
            buffer = glReadPixels(0, 0, 800, 600, GL_RGBA, GL_UNSIGNED_BYTE)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(600, 800, 4)
            image = np.flipud(image)  # Flip the image vertically
            pygame.image.save(pygame.Surface((800, 600), 0, image), "cube.png")
        
        clock.tick(60)


if __name__ == "__main__":
    cubeDemo()