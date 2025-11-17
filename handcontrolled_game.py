import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np

class HandControlledGame:
    def __init__(self):
        pygame.init()
        self.WIDTH = 800
        self.HEIGHT = 400
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Hand Controlled Game")
        
        self.player_size = 40
        self.player_x = 100
        self.player_y = self.HEIGHT - self.player_size - 10
        self.player_jump = False
        self.jump_count = 10
        self.obstacles = []
        self.obstacle_speed = 5
        self.score = 0
        
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.PURPLE = (128, 0, 128)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def detect_hand_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        is_hand_open = False
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4].y
                index_tip = hand_landmarks.landmark[8].y
                middle_tip = hand_landmarks.landmark[12].y
                ring_tip = hand_landmarks.landmark[16].y
                pinky_tip = hand_landmarks.landmark[20].y

                thumb_mid = hand_landmarks.landmark[3].y
                index_mid = hand_landmarks.landmark[6].y
                middle_mid = hand_landmarks.landmark[10].y
                ring_mid = hand_landmarks.landmark[14].y
                pinky_mid = hand_landmarks.landmark[18].y

                fingers_extended = [
                    thumb_tip < thumb_mid,
                    index_tip < index_mid,
                    middle_tip < middle_mid,
                    ring_tip < ring_mid,
                    pinky_tip < pinky_mid
                ]
                
                is_hand_open = sum(fingers_extended) >= 4
        
        return frame, is_hand_open

    def create_obstacle(self):
        if len(self.obstacles) == 0 or self.obstacles[-1][0] < self.WIDTH - 300:
            obstacle_width = 30
            obstacle_height = 50
            self.obstacles.append([self.WIDTH, self.HEIGHT - obstacle_height, obstacle_width, obstacle_height])

    def update_game_state(self, is_hand_open):
        if is_hand_open and not self.player_jump:
            self.player_jump = True
        
        if self.player_jump:
            if self.jump_count >= -10:
                neg = 1
                if self.jump_count < 0:
                    neg = -1
                self.player_y -= (self.jump_count ** 2) * 0.5 * neg
                self.jump_count -= 1
            else:
                self.player_jump = False
                self.jump_count = 10
                self.player_y = self.HEIGHT - self.player_size - 10

        for obstacle in self.obstacles:
            obstacle[0] -= self.obstacle_speed

        self.obstacles = [obs for obs in self.obstacles if obs[0] > -30]

        self.create_obstacle()

        player_rect = pygame.Rect(self.player_x, self.player_y, self.player_size, self.player_size)
        for obstacle in self.obstacles:
            if player_rect.colliderect(pygame.Rect(obstacle[0], obstacle[1], obstacle[2], obstacle[3])):
                return False

        self.score += 1
        return True

    def draw_game(self):
        self.screen.fill((0, 0, 128))

        pygame.draw.rect(self.screen, self.GREEN, 
                        (self.player_x, self.player_y, self.player_size, self.player_size))

        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.PURPLE, obstacle)

        score_text = self.font.render(f'Score: {self.score//10}', True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()

    def run(self):
        running = True
        game_active = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame, is_hand_open = self.detect_hand_gesture(frame)

            if game_active:
                game_active = self.update_game_state(is_hand_open)
                self.draw_game()
            else:
                self.screen.fill((0, 0, 0))
                game_over_text = self.font.render(f'Game Over! Score: {self.score//10}', True, self.WHITE)
                restart_text = self.font.render('Press SPACE to restart', True, self.WHITE)
                self.screen.blit(game_over_text, (self.WIDTH//2 - 100, self.HEIGHT//2 - 50))
                self.screen.blit(restart_text, (self.WIDTH//2 - 100, self.HEIGHT//2 + 50))
                pygame.display.flip()

                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    self.score = 0
                    self.obstacles = []
                    self.player_y = self.HEIGHT - self.player_size - 10
                    self.player_jump = False
                    game_active = True

            cv2.imshow('Hand Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            self.clock.tick(60)

        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    game = HandControlledGame()
    game.run()