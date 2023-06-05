from typing import Tuple, Union, Dict, List

import os
import math
import serial
import click
import yaml
import multiprocessing as mp
from multiprocessing.connection import Connection
import pygame
from pyvidplayer import Video
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
    speed = min_speed
    current = 0
    while True:
        # get time past since last iteration
        interval = time.time() - last_time
        last_time = time.time()

        if serial_connection.poll(
        ) and (new := float(serial_connection.recv())) != current:
            current = new
            speed = min_speed + current * (max_speed - min_speed)
            print(current, '\t', speed)
        seed += interval * speed

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

    while True:
        line = ser.readline()
        if line:
            try:
                current = map_range(
                    clamp(int(line.strip()), 0, 1023), 0.0, 1023.0, 0.0, 1.0
                    )
            except Exception:
                current = 0
            serial_connection.send(current)

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
@click.option('--plan',               help='plan of all events in a yaml file', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True)
@click.option('--window_dimensions',  help='width & height of the window, comma seperated', type=parse_dimensions, required=True)
@click.option('--bar_width',          help='width of the bar in the center of the frame', type=int, default=0)
@click.option('--video_dimensions',   help='width & height of the video, comma seperated (used for proportional scaling)', type=parse_dimensions, required=True)
@click.option('--min_speed',          help='minimum speed (when the arduino sends 0)', type=float, required=True)
@click.option('--max_speed',          help='maximum speed (when the arduino sends 1023)', type=float, required=True)
@click.option('--serial_port',        help='serial port path', type=str, required=True)
@click.option('--baudrate',           help='baudrate of the serial connection', type=int, required=True)
@click.option('--fps',                help='how many frames/sec to render', type=int, default=60, show_default=True)
@click.option('--seed',               help='starting seed', type=float, default=0.0, show_default=True)
@click.option('--text_bottom',        help='by how many pixels is the text offset from the bottom of the window', type=int, default=30)
@click.option('--text_color',         help='R, G, B color, comma seperated', type=parse_dimensions, default=(255, 255, 255))
@click.option('--font_size',          help='what font to use', type=int, default=72)
@click.option('--font',               help='what font size to use', type=str, required=False)
# yapf: enable
def stream(
        network: str,
        plan: str,
        window_dimensions: Tuple[int, int],
        bar_width: int,
        video_dimensions: Tuple[int, int],
        min_speed: float,
        max_speed: float,
        serial_port: str,
        baudrate: str,
        fps: int,
        seed: float,
        text_bottom: int,
        text_color: Tuple[int],
        font_size: int,
        font: str = None
    ) -> None:

    # setup pygame
    pygame.init()
    window = pygame.display.set_mode(window_dimensions)
    clock = pygame.time.Clock()

    # load font
    if not font in pygame.font.get_fonts():
        print(f'font {font} is invalid or not installed')
        font = None
    font_renderer = pygame.font.SysFont(font, font_size)

    # setup pipes
    image_conn_parent, image_conn_child = mp.Pipe()
    serial_conn_parent, serial_conn_child = mp.Pipe()

    # calculate dimensions & positions
    half_dimensions = (
        int((window_dimensions[0] - bar_width) / 2), window_dimensions[1]
        )
    gan_pos = {
        'width': min(half_dimensions),
        'height': min(half_dimensions),
        'x': int((half_dimensions[0] - min(half_dimensions)) / 2),
        'y': int((half_dimensions[1] - min(half_dimensions)) / 2)
        }
    video_pos = {
        'width':
        half_dimensions[0] if video_dimensions[0] >= video_dimensions[1] else
        int(video_dimensions[0] / video_dimensions[1] * half_dimensions[1]),
        'height':
        half_dimensions[1] if video_dimensions[0] <= video_dimensions[1] else
        int(video_dimensions[1] / video_dimensions[0] * half_dimensions[0]),
        'x':
        0 if video_dimensions[0] >= video_dimensions[1] else (
            int(
                half_dimensions[0] - video_dimensions[0] / video_dimensions[1]
                * half_dimensions[1]
                ) / 2
            ),
        'y':
        0 if video_dimensions[0] <= video_dimensions[1] else int((
            half_dimensions[1]
            - video_dimensions[1] / video_dimensions[0] * half_dimensions[0]
            ) / 2)
        }
    right_half_x = half_dimensions[0] + bar_width

    # load & process yaml file
    with open(plan) as file:
        events = yaml.safe_load(file)

    def filter_events(event):
        if 'time' in event and 'type' in event:
            if event['type'] == 'video':
                if not ('path' in event and os.path.exists(event['path'])):
                    print(
                        f'event: {event} from {plan} was discarded. "path" was either missing or invalid'
                        )
                    return False
            elif event['type'] == 'reposition':
                if not (
                    'gan' in event and
                    (event['gan'] == 'left' or event['gan'] == 'right')
                    ):
                    print(
                        f'event: {event} from {plan} was discarded. "gan" was either missing or invalid (must either be "left" or "right")'
                        )
                    return False
            elif event['type'] == 'text':
                if not ('content' in event and 'duration' in event):
                    print(
                        f'event: {event} from {plan} was discarded. "content" & "duration" attributes are mandatory for text events'
                        )
                    return False
            elif event['type'] == 'restart':
                pass
            else:
                print(
                    f'event: {event} from {plan} was discarded. type invalid'
                    )
                return False
        else:
            print(
                f'event: {event} from {plan} was discarded. "time" & "type" attributes are mandatory'
                )
            return False

        return True

    events = list(filter(filter_events, events))

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

    start_time = time.time()

    current_surface: pygame.Surface = None
    video: Video = None
    event_index = 0
    gan_position = 'left'
    current_text: Dict[str, Union[int, List[pygame.Surface]]] = None
    try:
        while True:
            # limit fps
            clock.tick(fps)

            # quit the loop when closed
            for pygame_event in pygame.event.get():
                if pygame_event.type == pygame.QUIT:
                    if video: video.close()
                    pygame.quit()
                    raise SystemExit
                elif pygame_event.type == pygame.KEYDOWN:
                    if pygame_event.key == pygame.K_f:
                        pygame.display.toggle_fullscreen()

            # get new gan image if avaliable
            if image_conn_parent.poll():
                pil_image: Image = image_conn_parent.recv()

                if pil_image.size != (gan_pos['width'], gan_pos['height']):
                    pil_image = pil_image.resize(
                        (gan_pos['width'], gan_pos['height'])
                        )

                current_surface = pilImageToSurface(pil_image)

            # get current event
            if (int(time.time() - start_time) == events[event_index]['time']):
                print(f'executing event {event_index}: {events[event_index]}')
                if events[event_index]['type'] == 'restart':
                    event_index = 0
                    start_time = time.time()
                else:
                    if events[event_index]['type'] == 'video':
                        if video: video.close()
                        video = Video(events[event_index]['path'])
                        video.set_volume(1.0)
                        video.set_size(
                            (video_pos['width'], video_pos['height'])
                            )
                    elif events[event_index]['type'] == 'reposition':
                        gan_position = events[event_index]['gan']
                    elif events[event_index]['type'] == 'text':
                        current_text = {
                            'images': [
                                font_renderer.render(line, True, text_color)
                                for line in events[event_index]
                                ['content'].split('\n')
                                ],
                            'duration':
                            events[event_index]['duration'],
                            'start':
                            time.time()
                            }

                    event_index += 1
                    if event_index >= len(events): event_index = 0

            # delete text if duration is up
            if current_text and time.time(
            ) - current_text['start'] > current_text['duration']:
                current_text = None

            # draw current surface if avaliable
            if current_surface:
                window.fill(0)
                window.blit(
                    current_surface,
                    (
                        gan_pos['x'] if gan_position == 'left' else
                        gan_pos['x'] + right_half_x,
                        gan_pos['y']
                        )
                    )
                if video:
                    video.draw(
                        window,
                        (
                            video_pos['x'] + right_half_x
                            if gan_position == 'left' else video_pos['x'],
                            video_pos['y']
                            ),
                        force_draw=True
                        )
                if current_text:
                    bottom = text_bottom
                    for image in reversed(current_text['images']):
                        window.blit(
                            image,
                            ((window_dimensions[0] - image.get_size()[0]) / 2,
                             window_dimensions[1] - bottom
                             - image.get_size()[1])
                            )
                        bottom += image.get_size()[1]
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