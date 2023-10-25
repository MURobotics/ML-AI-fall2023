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


rewardGates = [(VERTICALLINEMASK, 175, 275), (HORIZONTALLINEMASK , 180, 180),
               (VERTICALLINEMASK , 310, 25), (VERTICALLINEMASK , 470, 25),
               (HORIZONTALLINEMASK , 530, 130), (VERTICALLINEMASK , 430, 125),
               (HORIZONTALLINEMASK , 260, 240), (VERTICALLINEMASK , 430, 240),
               (HORIZONTALLINEMASK , 530, 425), (HORIZONTALLINEMASK , 425, 510),
               (VERTICALLINEMASK , 390, 325), (HORIZONTALLINEMASK , 275, 510),
               (HORIZONTALLINEMASK, 135,510),(HORIZONTALLINEMASK, 12,360),
                (HORIZONTALLINEMASK, 5,200), (VERTICALLINEMASK , 90, 25),
                (HORIZONTALLINEMASK, 90,250)]

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
        self.score = 0
    
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

    def move_direction(self, angle): #angle in degrees
        radians = math.radians(angle)
        vertical = self.vel * math.cos(radians) 
        horizontal = self.vel * math.sin(radians) 
        # trust in subtraction
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

    def get_x_y_orient(self):
        return [self.x, self.y, self.angle]
        
    def update_score(self, added_score):
        self.score += added_score

    def reset(self):
        self.vel = 0;
        self.angle = 0;
        self.x, self.y = self.START_POS;
        self.acceleration = 0.1;
        self.score = 0;

        return [self.x, self.y, self.angle];
        

class GameAI:
        def __init__(self):
            self.car = Car(3, 4)
            self.clock = pygame.time.Clock()
            self.images = [(DESERT, (0, 0)), (TRACK, (0, 0)), (FINISH, (88, 250)), (TRACK_BORDER, (0, 0))]
            self.numMapImages = len(self.images) #used to offset when getting reward gate masks later
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            self.gateIdx = 0 #gate index
            self.rewardGates = rewardGates
            pygame.display.set_caption("AI Driver!")
            self.reset()

        def drawRewardGates(self)->None:
            for gate in self.rewardGates:
                image = HORIZONTALLINE if gate[0] == HORIZONTALLINEMASK else VERTICALLINE
                self.images.append((image, (gate[1], gate[2])))

        def drawTrack(self, win, images, car):
            #This sets the reward gates images up correctly
            self.drawRewardGates() 
            for img, pos in self.images:
                win.blit(img, pos)
            
            car.draw(win)

        def checkCarCollision(self, rewardDecrement = 10):
            reward = 0
            if(self.car.car_collide(TRACK_BORDER_MASK) != None):
                self.car.bounce()
                reward = -rewardDecrement
            return reward
        
        def checkRewardGateCollisions(self, rewardFactor = 100):
            reward = 0
            #check if the gate at self.gateIdx has been passed by the car
            if(self.car.collide(self.images[self.numMapImages + self.gateIdx], self.rewardGates[self.gateIdx][1],self.rewardGates[self.gateIdx][2])):
                self.gateIdx += 1
                reward += (self.gateIdx + 1)*rewardFactor
            
            return reward
        
        def move_player(self, action):
            reward = -1
            carMove = 0
            for index in range(0, len(action)):
                if(action[index] == 1):
                    carMove = index
            moved = False

            #only 8 directions to move in, set angle according to the index
            self.car.move_direction(carMove * 45.0)
            
            return reward
        
        def play_move(self, action):
            #TODO this thing
            pass


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