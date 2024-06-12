import math
import logging
import numpy as np
from config import Config as cfg


# cp env

class Terminator:
    def __init__(self):
        self.timeout = False
        self.courseout = False
        self.goal = False
    
    def done(self):
        return any([self.courseout])
    
    def truncate(self):
        return any([self.timeout])

class Env:
    def __init__(self):
        self.time = 0
        self.world_width = 6   # 6m
        self.world_height = 4  # 4m
        self.screen_width = 1200
        self.screen_height = 800
        self.tolerance = np.concatenate([cfg.resolution * 10 , [0.01, 0.01]])
        #####  always start from the position out of tolerance
        # self.state = np.array([np.random.choice([np.random.uniform(-cfg.resolution[0]*50, -self.tolerance[0]),
        #                                          np.random.uniform(self.tolerance[0], cfg.resolution[0]*50)]),  # x
        #                        np.random.choice([np.random.uniform(-cfg.resolution[1]*50, -self.tolerance[1]),
        #                                          np.random.uniform(self.tolerance[1], cfg.resolution[1]*50)]),  # y
        #                        np.random.choice([np.random.uniform(-cfg.resolution[2]*50, -self.tolerance[2]),
        #                                          np.random.uniform(self.tolerance[2], cfg.resolution[2]*50)]),  # a
        #                        0.0,  # vx
        #                        0.0]) # wz
        
        #####  start from random position including inside the tolerance
        self.state = np.array([np.random.uniform(-cfg.resolution[0] * 50, cfg.resolution[0] * 50),  # x
                               np.random.uniform(-cfg.resolution[1] * 50, cfg.resolution[1] * 50),  # y
                               np.random.uniform(-cfg.resolution[2] * 50, cfg.resolution[2] * 50),  # a
                               0.0,  # vx
                               0.0]) # wz
        self.max_range = [cfg.resolution[0]*200, cfg.resolution[1]*200, cfg.resolution[2]*200, 0.3, 0.3]
        self.screen = None
        self.clock = None
        self.isopen = True
        self.terminator = Terminator()
        if cfg.mode == "x":
            self.state[1] = 0.0
            self.state[2] = 0.0
            self.tolerance[1] = None
            self.tolerance[2] = None
        elif cfg.mode == "xa":
            self.state[1] = 0.0
            self.tolerance[1] = None
        elif cfg.mode == "xya":
            pass
        else:
            raise ValueError(f"Invalid mode: {cfg.mode}")

    def _reward_x(self):
        x = self.state[0]
        reward = abs(x) * 10
        reward_max = self.max_range[0] * 10
        reward = reward_max - reward
        return reward
    
    def _reward_xa(self):
        x, a = self.state[0], self.state[2]
        n = cfg.resolution[0] / cfg.resolution[2]
        reward = abs(x) * 10 + abs(a) * 10 * n
        reward_max = self.max_range[0] * 10 + self.max_range[2] * 10 * n
        reward = reward_max - reward
        return reward
    
    def _reward_xya(self):
        pass

    @property
    def _reward(self):
        if cfg.mode == "x":
            reward = self._reward_x()
        elif cfg.mode == "xa":
            reward = self._reward_xa()
        elif cfg.mode == "xya":
            reward = self._reward_xya()

        if self.terminator.courseout:
            reward = 0

        return float(reward)

    @property
    def _done(self):
        return self.terminator.done()
    
    @property
    def _truncate(self):
        return self.terminator.truncate()

    @property
    def _info(self):
        return {"time": self.time}

    def _vel_control(self, action, vel):
        new_vel = vel + action * np.array([cfg.resolution[0], cfg.resolution[2]])
        return new_vel

    def _pose_transition(self, pose, vel, time_interval):
        theta = pose[2]
        vx, wz = vel
        if math.fabs(wz) < 1e-10:
            new_pose = pose + np.array([vx*math.cos(theta),
                                        vx*math.sin(theta),
                                        wz]) * time_interval
        else:
            new_pose = pose + np.array([vx / wz * (math.sin(theta + vx * time_interval) - math.sin(theta)),
                                        vx / wz * (-math.cos(theta + vx * time_interval) + math.cos(theta)),
                                        wz*time_interval])

        new_pose[2] = math.atan2(math.sin(new_pose[2]), math.cos(new_pose[2]))
        return new_pose
    
    def _render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            logging.error(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("linetrace game")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if self.state is None:
            return None

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        scale = self.screen_width / self.world_width
        screen_pose_x, screen_pose_y = self.state[0] * scale + self.screen_width / 2, self.state[1] * scale + self.screen_height / 2

        for i in range(0, self.screen_width, int(1 * scale)):
            gfxdraw.line(self.surf, i, 0, i, self.screen_height, (0, 0, 0))
        for i in range(0, self.screen_height, int(1 * scale)):
            gfxdraw.line(self.surf, 0, i, self.screen_width, i, (0, 0, 0))
        r = int(0.35 * scale)
        cam_r = int(0.15 * scale)
        screen_cam_pose_x = screen_pose_x + cam_r * math.cos(self.state[2])
        screen_cam_pose_y = screen_pose_y + cam_r * math.sin(self.state[2])
        gfxdraw.circle(self.surf, int(screen_pose_x), int(screen_pose_y), r, (255, 0, 0))
        # filled circle
        gfxdraw.circle(self.surf, int(screen_cam_pose_x), int(screen_cam_pose_y), 5, (255, 0, 0))
        gfxdraw.line(self.surf, int(screen_pose_x), int(screen_pose_y), int(screen_pose_x + r * math.cos(self.state[2])), int(screen_pose_y + r * math.sin(self.state[2])), (255, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))


        # write state on the screen
        font = pygame.font.Font(None, 25)
        _x = self.state[0] # self.state[0] + 0.15 * math.cos(self.state[2])
        _y = self.state[1] # self.state[1] + 0.15 * math.sin(self.state[2])
        text_x = font.render(f"x: {_x}", True, (0, 0, 0))
        text_y = font.render(f"y: {_y}", True, (0, 0, 0))
        text_a = font.render(f"a: {self.state[2]:.3f}", True, (0, 0, 0))
        text_vx = font.render(f"vx: {self.state[3]:.3f}", True, (0, 0, 0))
        text_wz = font.render(f"wz: {self.state[4]:.3f}", True, (0, 0, 0))
        text_time = font.render(f"time: {self.time:.3f}", True, (0, 0, 0))
        self.screen.blit(text_x, (20, 20))
        self.screen.blit(text_y, (20, 40))
        self.screen.blit(text_a, (20, 60))
        self.screen.blit(text_vx, (20, 80))
        self.screen.blit(text_wz, (20, 100))
        self.screen.blit(text_time, (20, 120))

        pygame.event.pump()
        self.clock.tick(cfg.fps)
        pygame.display.flip()

    def _update_terminator(self):
        # courseout
        self.terminator.courseout = \
            any([abs(state) > max_range \
                 for state, max_range in zip(self.state[:3], self.max_range[:3])])
        # timeout
        self.terminator.timeout = self.time > cfg.time_limit

    def reset(self):
        self.__init__()
        return self.state, {}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def step(self, action):
        if cfg.agent_type == "dac":
            action = np.array(cfg.actions[action])
        elif cfg.agent_type == "cac":
            if cfg.mode == "x":
                action = np.array([action, 0])
            elif cfg.mode == "xa":
                pass

        time_interval = 1 / cfg.fps
        self.time += time_interval
        vel = self._vel_control(action, self.state[3:5])
        pose = self._pose_transition(self.state[:3], vel, time_interval)
        self.state = np.concatenate([pose, vel])
        self._update_terminator()
        if cfg.render:
            self._render()

        if cfg.mode == "x":
            self.state[1] = 0.0
            self.state[2] = 0.0

        # return np.ndarray, float, bool, dict, dict 
        return self.state, self._reward, self._done, self._truncate, self._info, {}
