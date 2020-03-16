from itertools import cycle
import random
import sys

import pygame
from pygame.locals import *
from pygame.surfarray import array3d, pixels_alpha
from pygame.event import pump

import numpy as np

FPS = 30
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 100  # gap between upper and lower part of pipe
BASEY = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts
IMAGES, SOUNDS, HITMASKS = {}, {}, {}

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red player
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue player
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow player
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)

try:
    xrange
except NameError:
    xrange = range


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in xrange(image.get_width()):
        mask.append([])
        for y in xrange(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


class Flappyplayer():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy player')

    # numbers sprites for score display
    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    # game over sprite
    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    # message sprite for welcome screen
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    # base (ground) sprite
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()

    # sounds
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'

    SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + soundExt)
    SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + soundExt)
    SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + soundExt)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + soundExt)
    SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + soundExt)

    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES['player'] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES['pipe'] = (
        pygame.transform.flip(
            pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS['pipe'] = (
        getHitmask(IMAGES['pipe'][0]),
        getHitmask(IMAGES['pipe'][1]),
    )

    # hitmask for player
    HITMASKS['player'] = (
        getHitmask(IMAGES['player'][0]),
        getHitmask(IMAGES['player'][1]),
        getHitmask(IMAGES['player'][2]),
    )

    pipeVelX = -4

    # player velocity, max velocity, downward accleration, accleration on flap
    playerVelY = 0  # player's velocity along Y, default same as playerFlapped
    playerMaxVelY = 10  # max vel along Y, max descend speed
    playerMinVelY = -8  # min vel along Y, max ascend speed
    playerAccY = 1  # players downward accleration
    playerRot = 45  # player's rotation
    playerVelRot = 3  # angular speed
    playerRotThr = 20  # rotation threshold
    playerFlapAcc = -9  # players speed on flapping
    playerFlapped = False  # True when player flaps

    playerIndexGen = cycle([0, 1, 2, 1])

    baseShift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    player_hitmask = [pixels_alpha(image).astype(bool) for image in IMAGES['player']]
    pipe_hitmask = [pixels_alpha(image).astype(bool) for image in IMAGES['pipe']]

    def __init__(self):
        self.iter = self.player_index = self.score = 0
        self.player_width = IMAGES['player'][0].get_width()
        self.player_height = IMAGES['player'][0].get_height()
        self.pipe_width = IMAGES['pipe'][0].get_width()
        self.pipe_height = IMAGES['pipe'][0].get_height()

        self.player_x = int(SCREENWIDTH * 0.2)
        self.player_y = int((SCREENHEIGHT - self.player_height) / 2)

        self.base_x = 0
        self.base_y = SCREENHEIGHT * 0.79

        self.SCREENW = 288
        self.SCREENH = 512
        self.BASEY = SCREENHEIGHT * 0.79
        self.current_velocity_y = 0

        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()

        # list of upper pipes
        self.upperPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']},
        ]

        # list of lowerpipe
        self.lowerPipes = [
            {'x': SCREENWIDTH, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']},
        ]

        # self.current_velocity_y = 0
        self.is_flapped = False

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        # y of gap between upper and lower pipe
        gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
        gapY += int(BASEY * 0.2)
        pipeHeight = IMAGES['pipe'][0].get_height()
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE},  # lower pipe
        ]

    def checkCrash(self):
        """returns True if player collders with base or pipes."""
        if self.player_height + self.player_y + 1 >= self.base_y:
            return True
        playerRect = pygame.Rect(self.player_x, self.player_y, self.player_width, self.player_height)
        pipeW = IMAGES['pipe'][0].get_width()
        pipeH = IMAGES['pipe'][0].get_height()
        pipeBoxes = []
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipeRect = pipeBoxes.append(pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH))
            lPipeRect = pipeBoxes.append(pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH))
            if playerRect.collidelist(pipeBoxes) == -1:
                return False
            for i in range(2):
                croppedPlayerRect = playerRect.clip(pipeBoxes[i])
                minX1 = croppedPlayerRect.x - playerRect.x
                minY1 = croppedPlayerRect.y - playerRect.y
                minX2 = croppedPlayerRect.x - pipeBoxes[i].x
                minY2 = croppedPlayerRect.y - pipeBoxes[i].y
                if np.any(self.player_hitmask[self.player_index][minX1:minX1 + croppedPlayerRect.width,
                          minY1:minY1 + croppedPlayerRect.height] * self.pipe_hitmask[i][
                                                                    minX2:minX2 + croppedPlayerRect.width,
                                                                    minY2:minY2 + croppedPlayerRect.height]):
                    return True
        return False

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in xrange(rect.width):
            for y in xrange(rect.height):
                if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                    return True
        return False

    def next_frame(self, action):
        pump()
        reward = 0.1
        terminal = False
        if action == 1:
            self.current_velocity_y = self.playerFlapAcc
            self.is_flapped = True
            # SOUNDS['wing'].play()

        playerMidPos = self.player_x + IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
                SOUNDS['point'].play()
                reward = 1

        if (self.iter + 1) % 3 == 0:
            self.player_index = next(self.playerIndexGen)
            self.iter = 0
        # self.iter = (self.iter + 1) % 30
        self.base_x = -((-self.base_x + 100) % self.baseShift)

        # if self.playerRot > -90:
        #     self.playerRot -= self.playerVelRot

        if self.current_velocity_y < 10 and not self.is_flapped:
            self.current_velocity_y += 1
        if self.is_flapped:
            self.is_flapped = False
            # self.playerRot = 45
        self.player_y += min(self.current_velocity_y, self.player_y - self.current_velocity_y - self.player_height)
        if self.player_y < 0:
            self.player_y = 0

        # self.player_height = IMAGES['player'][self.player_index].get_height()

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if len(self.upperPipes) > 0 and 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if len(self.upperPipes) > 0 and self.upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)

        if self.checkCrash():
            terminal = True
            reward = -1
            self.__init__()

        SCREEN.blit(IMAGES['background'], (0, 0))
        SCREEN.blit(IMAGES['base'], (self.base_x, self.base_y))
        # visibleRot = self.playerRotThr
        # if self.playerRot <= self.playerRotThr:
        #     visibleRot = self.playerRot
        # playerSurface = pygame.transform.rotate(IMAGES['player'][self.player_index], visibleRot)
        SCREEN.blit(IMAGES['player'][self.player_index], (self.player_x, self.player_y))
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            SCREEN.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        image = array3d(pygame.display.get_surface())
        pygame.display.update()
        FPSCLOCK.tick(FPS)

        return image, reward, terminal
