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

INNER_TRACK_BORDER = scale_image(pygame.image.load("imgs/inner-track-border.png"), TRACK_SCALE_FACTOR)
INNER_TRACK_BORDER_MASK = pygame.mask.from_surface(INNER_TRACK_BORDER)

OUTER_TRACK_BORDER = scale_image(pygame.image.load("imgs/outer-track-border.png"), TRACK_SCALE_FACTOR)
OUTER_TRACK_BORDER_MASK = pygame.mask.from_surface(OUTER_TRACK_BORDER)

HORIZONTALLINE =  scale_image(pygame.image.load("imgs/horizontalline.png"),.1)
HORIZONTALLINEMASK = pygame.mask.from_surface(HORIZONTALLINE)
HORIZONTALLINE.fill(color="blue")

VERTICALLINE = scale_image(pygame.image.load("imgs/verticalline.png"), .2)
VERTICALLINEMASK = pygame.mask.from_surface(VERTICALLINE)
VERTICALLINE.fill(color="blue")

FINISH = pygame.image.load("imgs/finish.png")
FINISHMASK = pygame.mask.from_surface(FINISH)

CAR_SCALE_FACTOR = 0.6
RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), CAR_SCALE_FACTOR)
YELLOW_CAR = scale_image(pygame.image.load("imgs/yellow-car.png"), CAR_SCALE_FACTOR)

# NOTE: May have to change DPI to get screen to fit
WIDTH, HEIGHT = TRACK.get_width(), TRACK.get_height()

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
        self.rewardgate = 0
    
    def reset(self):
        self.vel = 0
        self.angle = 0
        self.x, self.y = self.START_POS
    
    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotational_vel
        elif right:
            self.angle -= self.rotational_vel
    
    def draw(self, win):
        blit_rotate_center(win, self.img, (self.x, self.y), self.angle)
        self.draw_rays(win,self,TRACK_BORDER_MASK,100)#Comment this out if you don't wanna see the rays
    
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
    def car_collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        # If poi (point of intersction) is none, no collision occured
        return poi
    
    #Gets the point of intersection between the car and the track border in four directions and returns those coordinates for each direction.
    def getWallPointOfIntersection(self,mask,x=0,y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = [int(self.x - x), int(self.y - y)]
        rightpoi = None
        leftpoi = None
        frontpoi = None
        rearpoi = None
        while(rightpoi == None):
            offset[0] +=1
            rightpoi = mask.overlap(car_mask,offset)
        while(leftpoi == None):
            offset[0] -=1
            leftpoi = mask.overlap(car_mask,offset)
        while(frontpoi == None):
            offset[0] = self.x
            offset[1] -=1
            frontpoi = mask.overlap(car_mask,offset)
        while(rearpoi == None):
            offset[0] = self.x
            offset[1] +=1
            rearpoi = mask.overlap(car_mask,offset)
        return rightpoi,leftpoi,frontpoi,rearpoi
    

    #gets the point of intersection between the car and the next reward gate returns that point. NOTE: Doesn't work and runs an infinite loop.
    def getRewardGatePointOfIntersection(self,mask,x=0,y=0):
        car_mask = pygame.mask.from_surface(self.img)
        # print(x)
        # print(y)
        offset = [int(self.x - x), int(self.y - y)]
        frontrewardpoi = None
        while(frontrewardpoi == None):
            offset[0] = self.x
            offset[1] -= 1
            frontrewardpoi = mask.overlap(car_mask,offset)
        return frontrewardpoi

    def bounce(self):
        self.vel = -self.vel
    
    def correctDirection(self):
        ray_length = 50
        
        correct_right_ray = self.cast_ray(0,OUTER_TRACK_BORDER_MASK,ray_length)
        correct_left_ray = self.cast_ray(math.pi,INNER_TRACK_BORDER_MASK,ray_length)
        wrong_right_ray = self.cast_ray(0,INNER_TRACK_BORDER_MASK,ray_length)
        wrong_left_ray = self.cast_ray(math.pi,OUTER_TRACK_BORDER_MASK,ray_length)
        
        if not correct_right_ray and wrong_right_ray:
            return 0
        
        if not correct_left_ray and wrong_left_ray:
            return 0
        
        return 1
    
    def distance_to_point(self, point):
        return math.sqrt((self.x - point[0]) ** 2 + (self.y - point[1]) ** 2)

    def cast_ray(self, direction, mask, max_length):#Can use this for casting rays which we will use to detect walls and/or reward gates.
        initDirection = 90
        radians = math.radians(self.angle + initDirection)
        dx = math.sin(radians + direction)
        dy = math.cos(radians + direction)
        
        x, y = self.x,self.y
        for _ in range(max_length):
            x += dx
            y += dy
            
            offset = (int(x),int(y))
            poi = mask.overlap(pygame.mask.from_surface(self.img),offset)
            
            if poi:
                distance = self.distance_to_point(offset)
                return offset, poi, distance
            
        return None
    
    def cast_rays(self, mask, max_length):
        right_ray = self.cast_ray(0, mask, max_length)
        left_ray = self.cast_ray(math.pi, mask, max_length)
        frontLeft_ray = self.cast_ray((math.pi*3)/4, mask, max_length)
        front_ray = self.cast_ray(math.pi / 2, mask, max_length)
        frontRight_ray = self.cast_ray(math.pi / 4, mask, max_length)
        rear_ray = self.cast_ray(-math.pi / 2, mask, max_length)

        return right_ray, left_ray, frontLeft_ray, front_ray, frontRight_ray, rear_ray
    
    def draw_rays(self,win, car, mask, max_length):
        rays = self.cast_rays(mask, max_length)
        i = 0
        for ray in rays:
            color = (255,0,0)
            if ray:
                # Draw the ray from the car's position to the point of intersection
                if i == 3:
                    color = (0,255,0)
                pygame.draw.line(win, color, (car.x + 10, car.y + 10), ray[0], 2)
            i += 1



class DriveGameAI:

    def __init__(self):
        self.car = Car(3, 4)
        self.clock = pygame.time.Clock()
        self.images = [(DESERT, (0, 0)), (TRACK, (0, 0)), (FINISH, (88, 250)), (TRACK_BORDER, (0, 0))]
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        self.horzGateInd = 0
        self.vertGateInd = 32
        self.lapcounter = 1
        self.walldistancearray = [0,0,0,0,0,0]
        self.arrayofrewardgatecoordinates = [(88,170), (89,95), (5,137),(8, 205),(8, 263),(9, 318),(15, 376), (47, 425),(78, 455), (124, 501), (163, 542), (276, 513), (278, 482),(278, 461), (279, 423),(428, 451), (429, 500), (431, 540),(538, 514),(537, 483),(537, 446),(538, 410),(536, 368),(537, 331),(274, 248),(535, 151),(538, 133),(534, 99),(177, 112),(177, 148),(175, 191), (176, 247),(87,24),(275, 531),(386, 336),(517, 531),(474, 246),(406, 247),(401, 164),(502, 165),(496, 18),(406, 19),(298, 18),(173, 282)]
        pygame.display.set_caption("AI Driver!")
        self.reset()
    
    def reset(self):
        self.car.reset()
        self.frame = 0
        self.horzGateInd = 0
        self.vertGateInd = 32
        self.lapcounter = 1
    
    def setWallDistances(self):
        rays = self.car.cast_rays(TRACK_BORDER_MASK,100)
        i = 0
        for ray in rays:
            if ray:
                self.walldistancearray[i] = int(ray[2])
            else:
                self.walldistancearray[i] = 100
            i += 1

    
    #draws reward gates to screen. First 30 coordinates are for horizontal lines. rest are for vertical.
    def drawRewardGates(self, win):
        i=0
        # rewardgatearray = [(HORIZONTALLINE,(88,150))]
        rewardgatearray = []
        #print(arrayofrewardgatecoordinates[0][0])
        # rewardgatemaskarray = [(HORIZONTALLINEMASK,(5,137))]
        rewardgatemaskarray = []
        for x,y in self.arrayofrewardgatecoordinates:
            if i <= 31:
                rewardgatearray.append((HORIZONTALLINE,(x,y)))
                rewardgatemaskarray.append((HORIZONTALLINEMASK,(x,y)))
            elif i > 31:
                rewardgatearray.append((VERTICALLINE,(x,y)))
                rewardgatemaskarray.append((VERTICALLINEMASK,(x,y)))
            win.blit(rewardgatearray[i][0],(rewardgatearray[i][1]))
            i+=1
        return rewardgatemaskarray, self.arrayofrewardgatecoordinates
    
    def draw(self, win, images, car):
        for img, pos in images:
            win.blit(img, pos)
    
        car.draw(win)
    
    def move_player(self, car, carMove):
        reward = 0
        # carMove = 0
        reward = 0
        moved = False
        # print(action)
        # for index in range(0, len(action)):
        #     if(action[index] == 1):
        #         carMove = index
        # moved = False

        if carMove  == 0:
            # print("LEFT")
            car.rotate(left=True)
        if carMove == 2:
            # print("RIGHT")
            car.rotate(right=True)
        if carMove ==1:
            moved=True
            car.move_forward()
            reward += 1
        # Change to if for cool speed glitch (hold both up and down arrow)
        # elif carMove >= 6:
        #     moved=True
        #     car.move_backward()
        
        if not moved:
            #reward -= 10
            car.reduce_speed()
        return reward
    
    def checkRewardGateCollisions(self, rewardgatemaskarray):
        reward = 0
        if(self.car.car_collide(rewardgatemaskarray[self.horzGateInd][0], rewardgatemaskarray[self.horzGateInd][1][0], rewardgatemaskarray[self.horzGateInd][1][1])):
            if(self.horzGateInd == 31):
                self.horzGateInd = 0
            else:
                self.horzGateInd += 1
            reward += 1000*(self.horzGateInd + 1) * self.lapcounter
    
        elif(self.car.car_collide(rewardgatemaskarray[self.vertGateInd][0], rewardgatemaskarray[self.vertGateInd][1][0], rewardgatemaskarray[self.vertGateInd][1][1])):
            if(self.vertGateInd == len(rewardgatemaskarray) - 1):
                self.vertGateInd = 32
                reward += 1000000000 * self.lapcounter
                print("lap completed")
                self.lapcounter += 1
                self.vertGateInd = 32
                self.horzGateInd = 0
            else:
                self.vertGateInd += 1
            reward += 1000*(self.vertGateInd  + 1) * self.lapcounter
                
            
        
        
        return reward
    
    def checkCarCollision(self):
        reward = 0
        if(self.car.car_collide(TRACK_BORDER_MASK) != None):
            self.car.bounce()
            reward = -1000
            self.reset()
        return reward
    
    def play_move(self, action):
        # Clock prevents faster than 60 FPS
        # self.clock.tick(FPS)
        self.frame += 1

        # Two things we return
        game_over = False
        reward = 0

        # Must reset if AI gets stuck
        if self.frame > 3000:
            game_over = True
            self.reset()

        # Updates new drawings/changes
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        reward += self.move_player(self.car, action)

        self.draw(self.display, self.images, self.car)


        rewardgatemaskarray, rewardgatemaskcoordinatearray = self.drawRewardGates(self.display)

        self.setWallDistances()

        reward += self.checkCarCollision()
        if (reward < 0 ):
            game_over = True
       
        
        reward += self.checkRewardGateCollisions(rewardgatemaskarray)
        
        self.car.move()



        return reward, game_over

