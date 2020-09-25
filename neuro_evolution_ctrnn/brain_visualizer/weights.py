import pygame
import numpy as np
import math
from typing import Tuple

from brain_visualizer import brain_visualizer


class Weights:

    @staticmethod
    def draw_weights(visualizer: "brain_visualizer.BrainVisualizer", start_pos_dict: dict, end_pos_dict: dict,
                     weight_matrix) -> None:
        for (start_neuron, end_neuron), weight in np.ndenumerate(weight_matrix):
            if weight != 0 and (
                    (weight > 0.0 and visualizer.positive_weights) or (weight < 0.0 and visualizer.negative_weights)):

                if visualizer.draw_threshold and abs(weight) < visualizer.draw_threshold:
                    continue

                start_pos = start_pos_dict[start_neuron]
                end_pos = end_pos_dict[end_neuron]

                if weight > 0.0:
                    weight_color = visualizer.color_positive_weight
                else:
                    weight_color = visualizer.color_negative_weight

                width = int(abs(weight)) + visualizer.weight_val

                if visualizer.weight_val == 0 and width < 1:
                    width = 1

                if visualizer.weights_direction:
                    # Angle of the line between both points to the x-axis
                    rotation = math.atan2((end_pos[1] - start_pos[1]), (end_pos[0] - start_pos[0]))

                    # Point, angle and length of the line for the endpoint of the arrow
                    trirad = 5 + width
                    arrow_length = (-1 * (visualizer.neuron_radius + trirad + 5))
                    arrow_end = (end_pos[0] + arrow_length * math.cos(rotation),
                                 end_pos[1] + arrow_length * math.sin(rotation))

                    if rotation != 0:
                        Weights.arrow(visualizer.screen, weight_color, weight_color, start_pos, arrow_end,
                                      trirad, width)
                else:
                    pygame.draw.line(visualizer.screen, weight_color, (int(start_pos[0]), int(start_pos[1])),
                                     (int(end_pos[0]), int(end_pos[1])), width)

    @staticmethod
    def arrow(screen: pygame.Surface, color: Tuple[int, int, int], tricolor: Tuple[int, int, int],
              start: Tuple[int, int], end: Tuple[int, int], trirad: int, width: int) -> None:
        if width >= 1:
            pygame.draw.line(screen, color, start, end, width)
            rotation = math.degrees(math.atan2(start[1] - end[1], end[0] - start[0])) + 90
            pygame.draw.polygon(screen, tricolor, (
                (
                    end[0] + trirad * math.sin(math.radians(rotation)),
                    end[1] + trirad * math.cos(math.radians(rotation))),
                (
                    end[0] + trirad * math.sin(math.radians(rotation - 120)),
                    end[1] + trirad * math.cos(math.radians(rotation - 120))), (
                    end[0] + trirad * math.sin(math.radians(rotation + 120)),
                    end[1] + trirad * math.cos(math.radians(rotation + 120)))))
