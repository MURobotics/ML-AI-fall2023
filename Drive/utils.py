import pygame

def scale_image(img: pygame.Surface, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

# Normally rotating images causes distortion since images are actually rectangles in PyGame
def blit_rotate_center(win, image, top_left, angle):
    # This rotates around top left (distortion)
    rotated_image = pygame.transform.rotate(image, angle)
    # Take rotated image and put it at center of old image
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft = top_left).center)
    win.blit(rotated_image, new_rect.topleft)