import pygame
import time
import math
from utils import scale_image, blit_rotate_center

DESERT = scale_image(pygame.image.load("imgs/desert.png"), 2)
# Edit this scale factor to fit own screen
TRACK_SCALE_FACTOR = 0.7
TRACK = scale_image(pygame.image.load("imgs/track.png"), TRACK_SCALE_FACTOR)
TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), TRACK_SCALE_FACTOR)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

HORIZONTALLINE =  scale_image(pygame.image.load("imgs/horizontalline.png"),.1)
HORIZONTALLINEMASK = pygame.mask.from_surface(HORIZONTALLINE)
VERTICALLINE = scale_image(pygame.image.load("imgs/verticalline.png"), .3)
#VERTICALLINE = pygame.transform.rotate(VERTICALLINE, 90)
VERTICALLINEMASK = pygame.mask.from_surface(VERTICALLINE)
#pygame.Surface.blit(LINE, TRACK)


rewardGates = [(HORIZONTALLINEMASK, 88, 150), (VERTICALLINEMASK , 120, 120)]

FINISH = pygame.image.load("imgs/finish.png")

CAR_SCALE_FACTOR = 0.6
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), CAR_SCALE_FACTOR)
YELLOW_CAR = scale_image(pygame.image.load("imgs/yellow-car.png"), CAR_SCALE_FACTOR)

# NOTE: May have to change DPI to get screen to fit
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()
WIN = pygame.display.set_mode((WIDTH, HEIGHT))

# Change Background Color to White (Obselete Now, just for info purposes)
WIN.fill((255, 255, 255))

pygame.display.set_caption("AI Driver!")

FPS = 60

class Car:
    IMG = RED_CAR
    START_POS = (140, 200)

    def __init__(self, max_vel, rotation_vel) -> None:
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotational_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
    
    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotational_vel
        elif right:
            self.angle -= self.rotational_vel
    
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
    
    # This function uses trig!
    def move(self):
        # Convert degrees to radians
        radians = math.radians(self.angle)
        # Use Trig
        vertical = self.vel * math.cos(radians) 
        horizontal = self.vel * math.sin(radians) 
        # Weird reasons for subtracting (just trust)
        self.x -= horizontal
        self.y -= vertical
    
    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        # self.acceleration = max(self.acceleration + 0.001, 0.15)
    
    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel/2)
        # self.acceleration = min(self.acceleration - 0.001, 0.1)
    
    def reduce_speed(self):
        if(self.vel >= 0):
            self.vel = max(self.vel - self.acceleration/2, 0)
        else:
            self.vel = min(self.vel + self.acceleration, 0)
    
    # Takes mask of object car could collide with (TRACK_BORDER_MASK) and its coordinates
    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        # If poi (point of intersction) is none, no collision occured
        return poi

    def bounce(self):
        self.vel = -self.vel
    

def draw(win, images, car):
    for img, pos in images:
        win.blit(img, pos)
    
    car.draw(win)

def stats(score):
    text_font = pygame.font.SysFont('arial', 12)


# Event Loop
run = True
clock = pygame.time.Clock()
images = [(DESERT, (0, 0)), (TRACK, (0, 0)), (FINISH, (88, 250)), (TRACK_BORDER, (0, 0)),(HORIZONTALLINE, (88,150)), (VERTICALLINE, (120,120))]
car = Car(3, 4)

def move_player(car):
    keys = pygame.key.get_pressed()
    moved = False

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        car.rotate(left=True)
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        car.rotate(right=True)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        moved=True
        car.move_forward()
    # Change to if for cool speed glitch (hold both up and down arrow)
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        moved=True
        car.move_backward()
    
    if not moved:
        car.reduce_speed()


while run:
    # Clock prevents faster than 60 FPS
    clock.tick(FPS)

    # Updates new drawings/changes
    pygame.display.flip()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
            break
    
    move_player(car)

    draw(WIN, images, car)
    
    if(car.collide(TRACK_BORDER_MASK) != None):
        car.bounce()
    elif(car.collide(HORIZONTALLINEMASK, 88, 150)!=None):
        car.bounce()
    elif(car.collide((VERTICALLINEMASK), 120,120)!=None):
        car.bounce()
    
    car.move()

    
    

pygame.quit()