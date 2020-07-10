import pygame
from neuro_evolution_ctrnn.brain_visualizer.color import Colour

class Neurons():
    def draw(self, myfont, positions, valueDict, minMax, hell, grau, grell, radius, matrix=False):
        for neuron in range(len(positions)):
            position = positions[neuron]
            pos_x = int(position[0])
            pos_y = int(position[1])

            if matrix == True:
                val = valueDict[neuron, neuron]
                colorVal = valueDict[neuron, neuron] / minMax
            else:
                val = valueDict[neuron]
                colorVal = valueDict[neuron] / minMax

            # Damit das Programm nicht abbricht wenn klipping range nicht passt
            # TODO: Das könnte man loggen
            if colorVal > 1:
                colorVal = 1
            if colorVal < -1:
                colorVal = -1

            if colorVal <= 0:
                # grau zu hell
                interpolierteFarbe = Colour.interpolateColor(grau, hell, abs(colorVal))
                textSurface = myfont.render(('%.5s' % val), False, self.black)
            else:
                # grau zu grün
                interpolierteFarbe = Colour.interpolateColor(grau, grell, colorVal)
                textSurface = myfont.render(('%.5s' % val), False, self.white)

            # Draw Circle and Text
            pygame.draw.circle(self.screen, interpolierteFarbe, (pos_x, pos_y), radius)
            self.screen.blit(textSurface, ((pos_x - 16), (pos_y - 7)))
