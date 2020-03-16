import argparse
import torch
from flappy import Flappyplayer
from util import pre_processing


def get_args():
    args = argparse.ArgumentParser("Play Flappy Bird with Trained Models")
    args.add_argument("--image_size", type=int, default=84)
    args.add_argument("--saved_path", type=str, default="trained_models")
    parse = args.parse_args()
    return parse


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        model = torch.load("{}/flappy_bird".format(opt.saved_path))
    else:
        torch.manual_seed(123)
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc: storage)

    model.eval()
    gameState = Flappyplayer()
    image, reward, terminal = gameState.next_frame(0)
    image = pre_processing(image[:gameState.SCREENW, :int(gameState.base_y)], opt.image_size,
                           opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction)

        nextImage, reward, terminal = gameState.next_frame(action)
        nextImage = pre_processing(nextImage[:gameState.SCREENW, :int(gameState.base_y)], opt.image_size,
                                   opt.image_size)
        nextImage = torch.from_numpy(nextImage)
        if torch.cuda.is_available():
            nextImage = nextImage.cuda()
        nextState = torch.cat((state[0, 1:, :, :], nextImage))[None, :, :, :]
        state = nextState


if __name__ == '__main__':
    arguments = get_args()
    test(arguments)
