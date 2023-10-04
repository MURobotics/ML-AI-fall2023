import pygame
import time
import math
from utils import scale_image, blit_rotate_center, sign
from Vector2d import Vector2D

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
    START_POS = Vector2D(140, 200)

    def __init__(self, max_vel, rotation_vel, max_grip = 0.05) -> None:
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = Vector2D(0, 0)
        self.rotational_vel = rotation_vel
        self.angle = 0
        self.position = Vector2D(self.START_POS.x, self.START_POS.y)
        self.acceleration = 0.1
        self.grip = max_grip

    def get_direction_vec(self):
        radians = -math.radians(self.angle + 90)
        return Vector2D(math.cos(radians), math.sin(radians))
    
    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotational_vel
        elif right:
            self.angle -= self.rotational_vel
    
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.position.x, self.position.y), self.angle)
    
    # This function uses trig!
    def move(self):
        self.position.x += self.vel.x
        self.position.y += self.vel.y
        self.stay_straight()
    
    def move_forward(self):
        direction = self.get_direction_vec()
        self.vel += self.acceleration * direction
        self.limit_speed()
    
    def move_backward(self):
        direction = self.get_direction_vec()
        self.vel -= self.acceleration * direction
        self.limit_speed()
    
    def reduce_speed(self):
        self.vel *= 0.98

    def limit_speed(self):
        velDirection = self.vel.getNormalised()
        directionSign = sign(self.vel.dot(self.get_direction_vec()))
        speed = self.vel.length
        if (directionSign > 0):
            self.vel -= velDirection * max(0, speed - self.max_vel) # max speed forwards
        else:
            self.vel -= velDirection * max(0, speed - self.max_vel * 0.5) # max speed backwards
    
    def stay_straight(self):
        direction = self.get_direction_vec()
        tangentVelocity = self.vel.dot(direction.normal())
        appliedGrip = sign(direction.normal().dot(self.vel)) * min(self.grip, abs(tangentVelocity))
        self.vel -= appliedGrip * direction.normal()

        # -0.08263467184899997 Vector2D {X:-0.8090169943749475, Y:0.587785252292473}
        # 6.866876550776764 Vector2D {X:-0.10452846326765347, Y:-0.9945218953682733}

    # Takes mask of object car could collide with (TRACK_BORDER_MASK) and its coordinates
    def has_collision(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.position.x - x), int(self.position.y - y))
        poi = mask.overlap(car_mask, offset)
        # If poi (point of intersection) is none, no collision occured
        return poi

    def bounce(self):
        self.vel = self.vel * -0.5 # Bounce car off wall by reversing velocity
        self.position += self.vel.getNormalised() * 5 # Move car out of wall so it doesn't reverse velocity twice 
    

def draw(win, images, car):
    for img, pos in images:
        win.blit(img, pos)
    
    car.draw(win)


# Event Loop
run = True
clock = pygame.time.Clock()
images = [(DESERT, (0, 0)), (TRACK, (0, 0)), (FINISH, (88, 250)), (TRACK_BORDER, (0, 0)),(HORIZONTALLINE, (88,150)), (VERTICALLINE, (120,120))]
car = Car(2.8, 3.4)

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
    
    if(car.has_collision(TRACK_BORDER_MASK) != None):
        car.bounce()
    # elif(car.has_collision(HORIZONTALLINEMASK, 88, 150)!=None):
    #     car.bounce()
    # elif(car.has_collision((VERTICALLINEMASK), 120,120)!=None):
    #     car.bounce()
    
    car.move()

    
    

pygame.quit()