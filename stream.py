from typing import Tuple, Union

import math
import serial
import click
import multiprocessing as mp
from multiprocessing.connection import Connection
import pygame
import time
import torch
import numpy as np
import dnnlib
import legacy
from PIL import Image as image
from PIL.Image import Image


def parse_dimensions(input: str) -> Tuple[str]:
    return tuple(int(dim.strip()) for dim in input.split(','))


def generate_frames(
        network: str,
        min_speed: float,
        max_speed: float,
        seed: float,
        image_connection: Connection,
        serial_connection: Connection
    ) -> None:
    # check for cuda
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        print('cuda is available.')
        device = torch.device('cuda')
    else:
        print('cuda is not available.')
        device = torch.device('cpu')
    print(f'device: "{device}"')

    # load model
    print(f'Loading networks from "{network}"...')
    with dnnlib.util.open_url(network) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # generate label
    label = torch.zeros([1, G.c_dim], device=device)

    last_time = time.time()
    seed_lower = None
    serial_input = 0
    speed = min_speed
    while True:
        # get time past since last iteration
        interval = time.time() - last_time
        last_time = time.time()

        if serial_connection.poll():
            current = float(serial_connection.recv())
            speed = min_speed + current * (max_speed - min_speed)
            print(current, '\t', speed)
        seed += interval * speed

        if serial_connection.poll():
            serial_input: int = serial_connection.recv()
            print(serial_input)  # TODO: do something with serial output

        # generate latent vectors
        if seed_lower != math.floor(seed):
            seed_lower = int(math.floor(seed))
            z_lower = torch.from_numpy(
                np.random.RandomState(seed_lower).randn(1, G.z_dim)
                ).to(device)

            seed_upper = int(math.ceil(seed))
            if seed_upper == seed_lower: seed_upper = seed_lower + 1
            z_upper = torch.from_numpy(
                np.random.RandomState(seed_upper).randn(1, G.z_dim)
                ).to(device)

        t = seed - seed_lower
        z = z_lower * (1 - t) + z_upper * t

        # generate image
        img: torch.Tensor = G(z, label, truncation_psi=1, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5
               + 128).clamp(0, 255).to(torch.uint8)

        pil_image = image.fromarray(img[0].cpu().numpy(), 'RGB')

        image_connection.send(pil_image)


def read_serial(
        serial_port: str, baudrate: int, serial_connection: Connection
    ) -> None:
    # setup serial
    ser = serial.Serial(serial_port, baudrate=baudrate, timeout=0.1)

    message_sent = False

    while True:
        line = ser.readline()
        if line:
            message_sent = False
            try:
                current = map_range(
                    clamp(int(line.strip()), 0, 1023), 0.0, 1023.0, 0.0, 1.0
                    )
            except Exception:
                current = 0

            serial_connection.send(current)
        elif not message_sent:
            print('problem with the serial connection')
            message_sent = True

        # reset the serial buffer
        ser.reset_input_buffer()


def clamp(
        x: Union[float, int], min: Union[float, int], max: Union[float, int]
    ) -> float:
    if x < min: return min
    if x > max: return max
    return x


def map_range(
        x: Union[float, int],
        in_min: Union[float, int],
        in_max: Union[float, int],
        out_min: Union[float, int],
        out_max: Union[float, int]
    ) -> float:
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def pilImageToSurface(pil_image: Image) -> pygame.Surface:
    return pygame.image.fromstring(
        pil_image.tobytes(), pil_image.size, pil_image.mode
        ).convert()


# yapf: disable
@click.command()
@click.option('--network',            help='model network pickle filename', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option('--window_dimensions',  help='width & height of the window, comma seperated', type=parse_dimensions, required=True)
@click.option('--min_speed',          help='minimum speed (when the arduino sends 0)', type=float, required=True)
@click.option('--max_speed',          help='maximum speed (when the arduino sends 1023)', type=float, required=True)
@click.option('--serial_port',        help='serial port path', type=str, required=True)
@click.option('--baudrate',           help='baudrate of the serial connection', type=int, required=True)
@click.option('--fps',                help='how many frames/sec to render', type=int, default=60, show_default=True)
@click.option('--seed',               help='starting seed', type=float, default=0.0, show_default=True)
# yapf: enable
def stream(
        network: str,
        window_dimensions: Tuple[int, int],
        min_speed: float,
        max_speed: float,
        serial_port: str,
        baudrate: str,
        fps: int,
        seed: float
    ) -> None:

    # setup pygame
    pygame.init()
    window = pygame.display.set_mode(window_dimensions)
    clock = pygame.time.Clock()

    # setup pipes
    image_conn_parent, image_conn_child = mp.Pipe()
    serial_conn_parent, serial_conn_child = mp.Pipe()

    image_dimensions = (min(window_dimensions), min(window_dimensions))
    # start processes
    generate_process = mp.Process(
        target=generate_frames,
        kwargs=({
            'network': network,
            'min_speed': min_speed,
            'max_speed': max_speed,
            'seed': seed,
            'image_connection': image_conn_child,
            'serial_connection': serial_conn_parent
            })
        )
    serial_process = mp.Process(
        target=read_serial,
        kwargs=({
            'serial_port': serial_port,
            'baudrate': baudrate,
            'serial_connection': serial_conn_child
            })
        )

    generate_process.start()
    serial_process.start()

    current_surface: pygame.Surface = None
    try:
        while True:
            # limit fps
            clock.tick(fps)

            # quite the loop when closed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise SystemExit
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()

            # get new surface if avaliable
            if image_conn_parent.poll():
                pil_image: Image = image_conn_parent.recv()

                if pil_image.size != image_dimensions:
                    pil_image = pil_image.resize(image_dimensions)

                current_surface = pilImageToSurface(pil_image)

            # draw current surface if avaliable
            if current_surface:
                window.fill(0)
                window.blit(
                    current_surface,
                    current_surface.get_rect(
                        center=tuple(el / 2 for el in window_dimensions)
                        )
                    )
            else:
                window.fill(0)

            # update window
            pygame.display.flip()
    except:
        if generate_process:
            generate_process.terminate()
            generate_process.join()
        if serial_process:
            serial_process.terminate()
            serial_process.join()
        if image_conn_parent: image_conn_parent.close()
        if image_conn_child: image_conn_child.close()
        if serial_conn_parent: serial_conn_parent.close()
        if serial_conn_child: serial_conn_child.close()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    stream()