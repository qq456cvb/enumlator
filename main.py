from PIL import ImageGrab
import numpy as np
import cv2
import math, sys
import win32gui
from matplotlib import pyplot as plt
import win32api
import win32con
import card
from card import Card, CardGroup
from collections import Counter
import time
import random


def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True


def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,x,y,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,x,y,0,0)


class Emulator:
    action_space = card.get_action_space()

    def __init__(self):
        self.last_cards = []
        self.cards = []
        self.click_pos = []
        self.extra_cards = []
        self.templates = [None] * len(Card.cards)
        self.templates_mini = [None] * len(Card.cards)
        self.templates_tiny = [None] * len(Card.cards)
        self.train()
        self.get_window('BlueStacks App Player for Windows (beta-1)')

    def train(self):
        for idx, card in enumerate(Card.cards):
            if card in ['*', '$']:
                self.templates[idx] = cv2.imread('joker.bmp')
            else:
                self.templates[idx] = cv2.imread('%s.bmp' % card)
            self.templates[idx] = cv2.cvtColor(self.templates[idx], cv2.COLOR_BGR2GRAY)
        for idx, card in enumerate(Card.cards):
            if card in ['*', '$']:
                self.templates_mini[idx] = cv2.imread('joker_mini.bmp')
            else:
                self.templates_mini[idx] = cv2.imread('%s_mini.bmp' % card)
            if self.templates_mini[idx] is not None:
                self.templates_mini[idx] = cv2.cvtColor(self.templates_mini[idx], cv2.COLOR_BGR2GRAY)
        for idx, card in enumerate(Card.cards):
            if card in ['*', '$']:
                self.templates_tiny[idx] = cv2.imread('joker_tiny.bmp')
            else:
                self.templates_tiny[idx] = cv2.imread('%s_tiny.bmp' % card)
            if self.templates_tiny[idx] is not None:
                self.templates_tiny[idx] = cv2.cvtColor(self.templates_tiny[idx], cv2.COLOR_BGR2GRAY)

    def get_mask(self):
        self.cards = self.parse_self_cards(self.get_window_img())
        # 1 valid; 0 invalid
        mask = np.zeros_like(self.action_space)
        for j in range(mask.size):
            if counter_subset(self.action_space[j], self.cards):
                mask[j] = 1
        mask = mask.astype(bool)
        if self.last_cards:
            for j in range(1, mask.size):
                if mask[j] == 1 and not CardGroup.to_cardgroup(self.action_space[j]).\
                        bigger_than(CardGroup.to_cardgroup(self.last_cards)):
                    mask[j] = False
        else:
            mask[0] = False
        return mask

    def step(self, i):
        intention = Emulator.action_space[i]
        print('intention:', end=' ')
        print(intention)
        pos = []
        cards = np.ma.array(self.cards, mask=np.zeros(len(self.cards)))
        for card in intention:
            idx = np.where(cards == card)[0][0]
            cards.mask[idx] = 1
            pos.append(self.click_pos[idx])
        for p in pos:
            click(p[0], p[1])

        if not intention:
            click(self.x + 600, self.y + 900)
        else:
            time.sleep(1)
            click(self.x + 1400, self.y + 900)

        time.sleep(3)
        next_cards = self.parse_next_cards(self.get_window_img())
        self.last_cards = self.parse_before_cards(self.get_window_img())
        if not self.last_cards:
            self.last_cards = next_cards
        # print(self.last_cards)

    def begin(self):
        # begin the game
        click(self.x + 1000, self.y + 900)

    def end(self):
        # click the summary window
        click(self.x + 1800, self.y + 200)

        time.sleep(1)
        # 三连败界面，傻逼作者
        if np.array_equal(self.get_window_img()[500, 1700, :], np.array([254, 231, 195])):
            click(self.x + 1800, self.y + 300)
            time.sleep(0.5)

    def prepare(self):
        # wait for shuffle
        time.sleep(5.5)

        self.extra_cards = self.parse_extra_cards(self.get_window_img())
        if not self.extra_cards:
            # I should choose
            if random.randint(1, 2) == 1:
                # do not call for lord
                click(self.x + 800, self.y + 900)
            else:
                # call for lord
                click(self.x + 1200, self.y + 900)
            time.sleep(0.25)
            self.extra_cards = self.parse_extra_cards(self.get_window_img())
            # no one calls for lord
            if not self.extra_cards:
                return self.prepare()

        time.sleep(5)
        next_cards = self.parse_next_cards(self.get_window_img())
        self.last_cards = self.parse_before_cards(self.get_window_img())
        if not self.last_cards:
            self.last_cards = next_cards
        print(self.last_cards)

    def parse_before_cards(self, img):
        cards = []
        method = eval('cv2.TM_CCOEFF_NORMED')
        for y in [560, 760]:
            x = 6
            while x < 1000:
                # cv2.rectangle(img, (6, 550), (1000, 800), (255, 0, 0))
                x += 1
                if np.array_equal(img[y, x, :], np.array([255, 255, 255])):
                    _, _, _, rect = cv2.floodFill(img.copy(), None, (x, y), (255, 0, 0), (0, 0, 0), (0, 0, 0))
                    sub_img_bgr = img[rect[1]:rect[1] + 48, rect[0]:rect[0] + rect[2]]
                    sub_img = cv2.cvtColor(sub_img_bgr, cv2.COLOR_BGR2GRAY)

                    max_response = 0.
                    max_j = -1
                    for (j, temp) in enumerate(self.templates_mini):
                        if temp is not None:
                            if sub_img.shape[1] < temp.shape[1]:
                                continue
                            res = cv2.matchTemplate(sub_img, temp, method)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            if max_val > max_response:
                                max_response = max_val
                                max_j = j
                    if max_response < 0.8:
                        continue
                    if Card.cards[max_j] in ['*', '$']:
                        if np.sum(sub_img_bgr[:, :, 2]) < np.sum(sub_img_bgr[:, :, 0]) + 1e4:
                            cards.append('*')
                            print('*', end=' ')
                        else:
                            cards.append('$')
                            print('$', end=' ')
                    else:
                        cards.append(Card.cards[max_j])
                        print(Card.cards[max_j], end=' ')
                    # print(max_response)
                    sys.stdout.flush()
                    x = rect[0] + rect[2] + 1
        return cards

    def parse_next_cards(self, img):
        cards = []
        method = eval('cv2.TM_CCOEFF_NORMED')
        for y in [560, 760]:
            x = 1080
            while x < 2100:
                # cv2.rectangle(img, (6, 550), (1000, 800), (255, 0, 0))
                x += 1
                if np.array_equal(img[y, x, :], np.array([255, 255, 255])):
                    _, _, _, rect = cv2.floodFill(img.copy(), None, (x, y), (255, 0, 0), (0, 0, 0), (0, 0, 0))
                    sub_img_bgr = img[rect[1]:rect[1] + 48, rect[0]:rect[0] + rect[2]]
                    sub_img = cv2.cvtColor(sub_img_bgr, cv2.COLOR_BGR2GRAY)

                    max_response = 0.
                    max_j = -1
                    for (j, temp) in enumerate(self.templates_mini):
                        if temp is not None:
                            if sub_img.shape[1] < temp.shape[1]:
                                continue
                            res = cv2.matchTemplate(sub_img, temp, method)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                            if max_val > max_response:
                                max_response = max_val
                                max_j = j
                    if max_response < 0.8:
                        continue
                    if Card.cards[max_j] in ['*', '$']:
                        if np.sum(sub_img_bgr[:, :, 2]) < np.sum(sub_img_bgr[:, :, 0]) + 1e4:
                            cards.append('*')
                            print('*', end=' ')
                        else:
                            cards.append('$')
                            print('$', end=' ')
                    else:
                        cards.append(Card.cards[max_j])
                        print(Card.cards[max_j], end=' ')
                    # print(max_response)
                    sys.stdout.flush()
                    x = rect[0] + rect[2] + 1
        return cards

    def parse_self_cards(self, img):
        cards = []
        self.click_pos = []
        method = eval('cv2.TM_CCOEFF_NORMED')
        x = 10
        while x < 1935:
            x += 1
            if np.array_equal(img[1007, x, :], np.array([255, 255, 255])):
                _, _, _, rect = cv2.floodFill(img.copy(), None, (x, 1007), (255, 0, 0), (0, 0, 0), (0, 0, 0))
                sub_img_bgr = img[rect[1]:rect[1] + 75, rect[0]:rect[0] + rect[2]]
                sub_img = cv2.cvtColor(sub_img_bgr, cv2.COLOR_BGR2GRAY)
                # _, sub_img = cv2.threshold(sub_img, 127, 255, cv2.THRESH_BINARY)
                # cv2.imwrite('haha.bmp', sub_img)
                # plt.imshow(sub_img_bgr), plt.show()
                # cv2.imshow('test', sub_img_bgr)
                # cv2.waitKey(0)
                # _, sub_img = cv2.threshold(sub_img, 127, 255, cv2.THRESH_BINARY)

                max_response = 0.
                max_j = -1
                for (j, temp) in enumerate(self.templates):
                    if temp is not None:
                        if sub_img.shape[1] < temp.shape[1]:
                            continue
                        res = cv2.matchTemplate(sub_img, temp, method)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        if max_val > max_response:
                            max_response = max_val
                            max_j = j
                # if max_response < 0.8:
                #     continue
                if Card.cards[max_j] in ['*', '$']:
                    if np.sum(sub_img_bgr[:, :, 2]) < np.sum(sub_img_bgr[:, :, 0]) + 1e4:
                        cards.append('*')
                        print('*', end=' ')
                    else:
                        cards.append('$')
                        print('$', end=' ')
                else:
                    cards.append(Card.cards[max_j])
                    print(Card.cards[max_j], end=' ')
                # print(max_response)
                sys.stdout.flush()
                self.click_pos.append((self.x + rect[0] + int(rect[2] / 2), self.y + rect[1] + int(rect[3] / 2)))
                x = rect[0] + rect[2]
        print("")
        return cards

    def parse_extra_cards(self, img):
        cards = []
        method = eval('cv2.TM_CCOEFF_NORMED')
        x = 820
        while x < 1140:
            x += 1
            if np.array_equal(img[750, x, :], np.array([255, 255, 255])):
                _, _, _, rect = cv2.floodFill(img.copy(), None, (x, 750), (255, 0, 0), (0, 0, 0), (0, 0, 0))
                sub_img_bgr = img[rect[1]:rect[1] + 75, rect[0]:rect[0] + rect[2]]
                sub_img = cv2.cvtColor(sub_img_bgr, cv2.COLOR_BGR2GRAY)
                # _, sub_img = cv2.threshold(sub_img, 127, 255, cv2.THRESH_BINARY)
                # cv2.imwrite('haha.bmp', sub_img)
                # plt.imshow(sub_img_bgr), plt.show()
                # cv2.imshow('test', sub_img_bgr)
                # cv2.waitKey(0)
                # _, sub_img = cv2.threshold(sub_img, 127, 255, cv2.THRESH_BINARY)

                max_response = 0.
                max_j = -1
                for (j, temp) in enumerate(self.templates_tiny):
                    if temp is not None:
                        if sub_img.shape[1] < temp.shape[1]:
                            continue
                        res = cv2.matchTemplate(sub_img, temp, method)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                        if max_val > max_response:
                            max_response = max_val
                            max_j = j
                # if max_response < 0.8:
                #     continue
                if Card.cards[max_j] in ['*', '$']:
                    if np.sum(sub_img_bgr[:, :, 2]) < np.sum(sub_img_bgr[:, :, 0]) + 1e4:
                        cards.append('*')
                        print('*', end=' ')
                    else:
                        cards.append('$')
                        print('$', end=' ')
                else:
                    cards.append(Card.cards[max_j])
                    print(Card.cards[max_j], end=' ')
                # print(max_response)
                sys.stdout.flush()
                x = rect[0] + rect[2]
        print("")
        return cards

    def get_window(self, name):
        hwnd = win32gui.FindWindow(None, name)
        rect = win32gui.GetWindowRect(hwnd)
        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2] - self.x
        self.h = rect[3] - self.y
        # print("Window %s:" % win32gui.GetWindowText(hwnd))
        # print("\tLocation: (%d, %d)" % (x, y))
        # print("\t    Size: (%d, %d)" % (w, h))

    def get_window_img(self):
        img = ImageGrab.grab(bbox=(self.x, self.y, self.x + self.w, self.y + self.h))
        frame = np.array(img)
        frame = frame[:, :, [2, 1, 0]]
        return frame


if __name__ == '__main__':
    # emulator = Emulator()
    # i = 1
    # while cv2.imread('test%d.bmp' % i) is not None:
    #     i += 1
    # cv2.imwrite('test27.bmp', emulator.get_window_img())
    # img = emulator.get_window_img()
    # print(img[500, 1700, :])
    # assert np.array_equal(img[500, 1700, :], np.array([254, 231, 195]))
    # img = cv2.imread('test26.bmp')
    # h, w, _ = img.shape
    # for i in range(13):
    #     cv2.rectangle(img, (int(w / 13 * i), 0), (int(w / 13 * (i+1)), 50), (255, 0, 0))
    # cv2.rectangle(img, (1080, 560), (2100, 760), (255, 0, 0))
    # cv2.imshow('test', img)
    # cv2.waitKey(0)

    emulator = Emulator()
    for i in range(5):
        emulator.begin()
        emulator.prepare()
        while True:
            mask = emulator.get_mask()
            if np.sum(mask) == 0:
                break
            valid_actions = np.take(np.arange(len(Emulator.action_space)), mask.nonzero())
            valid_actions = valid_actions.reshape(-1)
            # valid_p = np.take(policy[0], mask.nonzero())
            # valid_p = valid_p / np.sum(valid_p)
            # valid_p = valid_p.reshape(-1)
            a = np.random.choice(valid_actions)
            emulator.step(a)
        emulator.end()
    # cv2.imwrite('test17.bmp', emulator.get_window_img())
    # print(emulator.parse_extra_cards(img))
    # print(emulator.parse_next_cards(img))
    # emulator.parse_self_cards(emulator.get_window_img())
    # for p in emulator.click_pos:
    #     click(p[0], p[1])
    # cv2.waitKey(0)
    # while(True):
    #     frame = emulator.get_window_img()
    #
    #     # emulator.parse_cards(frame)
    #
    #     # height, width = frame.shape[:2]
    #     # cv2.imwrite('./test3.bmp', frame)
    #     # frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    #     cv2.imshow("test", frame)
    #     cv2.waitKey(0)
    #     break
    # cv2.destroyAllWindows()
