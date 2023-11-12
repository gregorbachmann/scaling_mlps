import os
from urllib.request import urlretrieve
import progressbar


default_checkpoints = {
    'B_12-Wi_1024_res_64_in21k': 'B-12_Wi-1024_res_64_imagenet21_epochs_800',
    'B_12-Wi_512_res_64_in21k': 'B-12_Wi-512_res_64_imagenet21_epochs_600',
    'B_6-Wi_1024_res_64_in21k': 'B-6_Wi-1024_res_64_imagenet21_epochs_800',
    'B_6-Wi_512_res_64_in21k': 'B-6_Wi-512_res_64_imagenet21_epochs_800',
    'B_12-Wi_1024_res_64_in21k_cifar10': 'B-12_Wi-1024_res_64_cifar10_epochs_20',
    'B_12-Wi_1024_res_64_in21k_cifar100': 'B-12_Wi-1024_res_64_cifar100_epochs_40',
    'B_12-Wi_1024_res_64_in21k_imagenet': 'B-12_Wi-1024_res_64_imagenet_epochs_50',
    'B_12-Wi_512_res_64_in21k_cifar10': 'B-12_Wi-512_res_64_cifar10_epochs_20',
    'B_12-Wi_512_res_64_in21k_cifar100': 'B-12_Wi-512_res_64_cifar100_epochs_20',
    'B_12-Wi_512_res_64_in21k_imagenet': 'B-12_Wi-512_res_64_imagenet_epochs_20',
    'B_6-Wi_512_res_64_in21k_cifar10': 'B-6_Wi-512_res_64_cifar10_epochs_20',
    'B_6-Wi_512_res_64_in21k_cifar100': 'B-6_Wi-512_res_64_cifar100_epochs_20',
    'B_6-Wi_512_res_64_in21k_imagenet': 'B-6_Wi-512_res_64_imagenet_epochs_20',
    'B_6-Wi_1024_res_64_in21k_cifar10': 'B-6_Wi-1024_res_64_cifar10_epochs_20',
    'B_6-Wi_1024_res_64_in21k_cifar100': 'B-6_Wi-1024_res_64_cifar100_epochs_20',
    'B_6-Wi_1024_res_64_in21k_imagenet': 'B-6_Wi-1024_res_64_imagenet_epochs_20'
}

weight_urls = {
    'B-12_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1rcV8RXij_kW9X2zSLNyNOTO_bKUjE0cJ&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
        'https://drive.google.com/uc?id=1sL9j_4FFeBTWTzuRFbHLfwrmPVLXhUtW&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1cmO3QSz_hfHtyzkUOZnPmXPxIE2YzEbf&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1QV3a99UT8llfh9zDDuNKWDH_5c6S_YT5&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1GyxuewoOzMRhzEOyUrIBLQzquc-QEYNV&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=1LNqC58cSwtuDr-C4bk1O3GA_vAWls-UH&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=1MVebvnSGL02k_ql1gUCjh4quGqM9RM4F&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar10_epochs_20':   
        'https://drive.google.com/uc?id=1F1NvoOsYCgsn1GOZcwsoToOtsw-9Aw1v&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar100_epochs_20':   
        'https://drive.google.com/uc?id=1KIULehrqOyxIkZj0HiqowNmBy4Ye1EQ2&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet_epochs_20':   
        'https://drive.google.com/uc?id=1f0ZYzB_XujX8hDcEn_J6iWvw1meJ4Cbg&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar10_epochs_20':   
        'https://drive.google.com/uc?id=1Tyd5CkROPCMQybnrZ_o1wiAW7VoK6AfJ&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar100_epochs_20':   
        'https://drive.google.com/uc?id=1FrRb78bjun6QGbbH-pCWDaaE_8LWW785&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet_epochs_20':   
        'https://drive.google.com/uc?id=115Lks211vx1at2dWn3JtQ57EZ7eNAVP4E&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1VjHgjheSm_w7xPtheEmY5kV_KE4-38zQ&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar100_epochs_20':
        'https://drive.google.com/uc?id=1iK3t20-GS_Vs-_Q3ZexSiCfjGJ3IaPC2&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet_epochs_20':
        'https://drive.google.com/uc?id=1iK3t20-GS_Vs-_Q3ZexSiCfjGJ3IaPC2&export=download&confirm=t&uuid'
}

config_urls = {
    'B-12_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=1envpLKUa9LhUlp2dLIL8Jb8447wwpXF0&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',

}


def download(name, checkpoint_path):
    weight_url = weight_urls[name]
    config_url = config_urls[name]

    weight_path = checkpoint_path + name + '_weights'
    config_path = checkpoint_path + name + '_config'
    weight_exists = os.path.isfile(weight_path)
    config_exists = os.path.isfile(config_path)

    if not weight_exists:
        print('Downloading weights...')
        urlretrieve(weight_url, weight_path, show_progress)
    else:
        print('Weights already downloaded')
    if not config_exists:
        urlretrieve(config_url, config_path)

    return weight_path, config_path


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None