import os
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import matplotlib.pyplot as plt

figuresFolder = "/Users/qxu5/Documents/AAPM2024/manufigures"
topicFolder = os.path.join(figuresFolder, "BeamletDosecalc")
if not os.path.isdir(topicFolder):
    os.mkdir(topicFolder)

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
            buffer = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(600, 800, 3)
            image = np.flipud(image)
            plt.imsave(os.path.join(topicFolder, "cube.png"), image)
            break
        
        clock.tick(60)
    
theta_value = np.pi / 16
nPhi = 8
edgePoints = 3
colorSurfaceX = (0, 1, 0, 0.25)
colorSurfaceY = (0, 0, 1, 0.25)
colorSurfaceZ = (1, 0, 1, 0.25)
colorLineX = (0, 1, 0, 1)
colorLineY = (0, 0, 1, 1)
colorLineZ = (0, 1, 1, 1)
yRotateAngle = 30  # degrees
xRotateAngle = 30  # degrees


def drawLattice():
    halfEdgePoints = edgePoints / 2
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # firstly, draw the surfaces
    glBegin(GL_QUADS)
    glColor4fv(colorSurfaceX)
    for i in range(edgePoints + 1):
        xDisplacement = (i - halfEdgePoints)
        point1 = (xDisplacement, halfEdgePoints, halfEdgePoints)
        point2 = (xDisplacement, -halfEdgePoints, halfEdgePoints)
        point3 = (xDisplacement, -halfEdgePoints, -halfEdgePoints)
        point4 = (xDisplacement, halfEdgePoints, -halfEdgePoints)
        glVertex3fv(point1)
        glVertex3fv(point2)
        glVertex3fv(point3)
        glVertex3fv(point4)

    glColor4fv(colorSurfaceY)
    for i in range(edgePoints + 1):
        yDisplacement = (i - halfEdgePoints)
        point1 = (halfEdgePoints, yDisplacement, halfEdgePoints)
        point2 = (-halfEdgePoints, yDisplacement, halfEdgePoints)
        point3 = (-halfEdgePoints, yDisplacement, -halfEdgePoints)
        point4 = (halfEdgePoints, yDisplacement, -halfEdgePoints)
        glVertex3fv(point1)
        glVertex3fv(point2)
        glVertex3fv(point3)
        glVertex3fv(point4)

    glColor4fv(colorSurfaceZ)
    for i in range(edgePoints + 1):
        zDisplacement = (i - halfEdgePoints)
        point1 = (halfEdgePoints, halfEdgePoints, zDisplacement)
        point2 = (-halfEdgePoints, halfEdgePoints, zDisplacement)
        point3 = (-halfEdgePoints, -halfEdgePoints, zDisplacement)
        point4 = (halfEdgePoints, -halfEdgePoints, zDisplacement)
        glVertex3fv(point1)
        glVertex3fv(point2)
        glVertex3fv(point3)
        glVertex3fv(point4)
    glEnd()

    # secondly, draw the lines
    glBegin(GL_LINES)
    glColor4fv(colorLineX)
    # for i in range(edgePoints - 1, edgePoints + 1):
    for i in range(edgePoints + 1):
        for j in range(edgePoints + 1):
            point1 = (-halfEdgePoints, i-halfEdgePoints, j-halfEdgePoints)
            point2 = (halfEdgePoints, i-halfEdgePoints, j-halfEdgePoints)
            glVertex3fv(point1)
            glVertex3fv(point2)

    glColor4fv(colorLineY)
    for i in range(edgePoints + 1):
        for j in range(edgePoints + 1):
            # point1 = (i-halfEdgePoints, halfEdgePoints-1, j-halfEdgePoints)
            point1 = (i-halfEdgePoints, -halfEdgePoints, j-halfEdgePoints)
            point2 = (i-halfEdgePoints, halfEdgePoints, j-halfEdgePoints)
            glVertex3fv(point1)
            glVertex3fv(point2)

    glColor4fv(colorLineZ)
    for i in range(edgePoints + 1):
        # for j in range(edgePoints - 1, edgePoints + 1):
        for j in range(edgePoints + 1):
            point1 = (i-halfEdgePoints, j-halfEdgePoints, -halfEdgePoints)
            point2 = (i-halfEdgePoints, j-halfEdgePoints, halfEdgePoints)
            glVertex3fv(point1)
            glVertex3fv(point2)
    glEnd()


rayColor1 = (1, 0, 0, 1)
rayThickness = 3

def oneThetaCCCS():
    """
    This function plots the diagram of one CCCS convolution angle
    """
    pygame.init()
    display = (600, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    halfEdgePoints = edgePoints / 2

    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glTranslatef(0, 0.2, -7)
    glClearColor(1.0, 1.0, 1.0, 1.0)

    glEnable(GL_LINE_SMOOTH)
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)

    clock = pygame.time.Clock()
    
    glRotatef(30, 0, 1, 0)
    xAxisAfterRotation = (np.cos(xRotateAngle * np.pi / 180), 0, np.sin(xRotateAngle * np.pi / 180))
    glRotatef(30, *xAxisAfterRotation)
    for j in range(2):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        drawLattice()

        # varying theta
        phiAngle = 0  # rad
        sourcePoint = np.array((0, edgePoints, 0))
        for k in range(-4, 5):
            thetaAngle = (k + 0.5) * np.pi / 8  # rad
            direction = np.array((np.sin(thetaAngle) * np.cos(phiAngle), -np.cos(thetaAngle),
                np.sin(thetaAngle) * np.sin(phiAngle)))
            
            yPointsList = []
            for ll in range(edgePoints + 1):
                scale = - ll / direction[1]
                point = sourcePoint + scale * direction
                yPointsList.append((scale, point))
            print(yPointsList)
            return


        pygame.display.flip()
        clock.tick(60)


    buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(buffer, dtype=np.uint8).reshape((display[1], display[0], 3))
    image = np.flipud(image)
    plt.imsave(os.path.join(topicFolder, "convRay1.png"), image)


if __name__ == "__main__":
    # cubeDemo()
    oneThetaCCCS()