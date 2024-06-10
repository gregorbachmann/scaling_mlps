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
        'https://drive.usercontent.google.com/download?id=1rcV8RXij_kW9X2zSLNyNOTO_bKUjE0cJ&export=download&authuser=0&confirm=t&uuid=72ba7ef7-5c0e-43a8-8538-c78a7b6ae34c&at=APZUnTVYImDEDtOncjUjlRW2Fa-v%3A1718049334362',
    'B-12_Wi-512_res_64_imagenet21_epochs_600':
        'https://drive.usercontent.google.com/download?id=1sL9j_4FFeBTWTzuRFbHLfwrmPVLXhUtW&export=download&authuser=0&confirm=t&uuid=91299093-c2dc-4538-93fa-d34be798cedc&at=APZUnTUEplQUhKAe6zbjUuFWUGiV%3A1718049319992',
    'B-6_Wi-1024_res_64_imagenet21_epochs_800':
        'https://drive.usercontent.google.com/download?id=1cmO3QSz_hfHtyzkUOZnPmXPxIE2YzEbf&export=download&authuser=0&confirm=t&uuid=c102f0ec-18b9-496a-b615-819513501d65&at=APZUnTX6iqbWKmcVQzv4nf04efor%3A1718049304706',
    'B-6_Wi-512_res_64_imagenet21_epochs_800':
        'https://drive.usercontent.google.com/download?id=1QV3a99UT8llfh9zDDuNKWDH_5c6S_YT5&export=download&authuser=0&confirm=t&uuid=fa3e3e51-9eae-4f4c-9c88-f9882258160c&at=APZUnTWSzjI5fY70cc3I1t_E3nv1%3A1718049288621',
    'B-12_Wi-1024_res_64_cifar10_epochs_20':
        'https://drive.usercontent.google.com/download?id=1GyxuewoOzMRhzEOyUrIBLQzquc-QEYNV&export=download&authuser=0&confirm=t&uuid=02337a36-362b-41bc-8b66-c1e8737c6729&at=APZUnTV0pOpn9aeIkKng_OtiRw0l%3A1718049274446',
    'B-12_Wi-1024_res_64_cifar100_epochs_40':
        'https://drive.usercontent.google.com/download?id=1LNqC58cSwtuDr-C4bk1O3GA_vAWls-UH&export=download&authuser=0&confirm=t&uuid=37ee7032-ec34-4414-ac3b-fcb8a3f5e17d&at=APZUnTXsPCgnt2__IQ7fScHpXmcX%3A1718049262372',
    'B-12_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.usercontent.google.com/download?id=1MVebvnSGL02k_ql1gUCjh4quGqM9RM4F&export=download&authuser=0&confirm=t&uuid=4118f07e-ffdd-4b74-9b74-2508ffcc2b02&at=APZUnTWAAiNwrzrTzDm3Sl3MtzMF%3A1718049247748',
    'B-12_Wi-512_res_64_cifar10_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1F1NvoOsYCgsn1GOZcwsoToOtsw-9Aw1v&export=download&authuser=0&confirm=t&uuid=899cb74b-2bce-4b51-81ab-2df63af3dcbe&at=APZUnTWAC-eENGH6rRWchnMHsSBm%3A1718049232656',
    'B-12_Wi-512_res_64_cifar100_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1KIULehrqOyxIkZj0HiqowNmBy4Ye1EQ2&export=download&authuser=0&confirm=t&uuid=bf208699-bbf3-4ad3-9bb6-d61eec237265&at=APZUnTXwvPbxLngt1wCVCVNriiXA%3A1718049215282',
    'B-12_Wi-512_res_64_imagenet_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1f0ZYzB_XujX8hDcEn_J6iWvw1meJ4Cbg&export=download&authuser=0&confirm=t&uuid=90d9b1d0-fd5f-468e-b637-53afa56d3f22&at=APZUnTXFslB7n2WqcFxsmxZzNqoB%3A1718049045796',
    'B-6_Wi-1024_res_64_cifar10_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1Tyd5CkROPCMQybnrZ_o1wiAW7VoK6AfJ&export=download&authuser=0&confirm=t&uuid=7534d430-40ab-4475-a862-9413499b0f79&at=APZUnTXX2ioldX_5JCXDt0nP3pCu%3A1718049195583',
    'B-6_Wi-1024_res_64_cifar100_epochs_20':   
        'https://drive.usercontent.google.com/download?id=1FrRb78bjun6QGbbH-pCWDaaE_8LWW785&export=download&authuser=0&confirm=t&uuid=b2c5459c-ede5-4ba4-97dc-e7a247cfba6a&at=APZUnTWa7Uha96h-6FxJosR1b2F0%3A1718048945010',
    'B-6_Wi-1024_res_64_imagenet_epochs_20':   
        'https://drive.google.com/uc?id=115Lks211vx1at2dWn3JtQ57EZ7eNAVP4E&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.usercontent.google.com/download?id=1VjHgjheSm_w7xPtheEmY5kV_KE4-38zQ&export=download&authuser=0&confirm=t&uuid=71fbd376-79f4-43db-bd4a-381255571319&at=APZUnTUJN8LEutgI-L0oVjbG3df3%3A1718048695392',
    'B-6_Wi-512_res_64_cifar100_epochs_20':
        'https://drive.usercontent.google.com/download?id=1iK3t20-GS_Vs-_Q3ZexSiCfjGJ3IaPC2&export=download&authuser=0&confirm=t&uuid=0196232e-f83d-4c9d-921e-857db8848725&at=APZUnTV3aw0EOkJS4SEIo4XToVT4%3A1718048904050',
    'B-6_Wi-512_res_64_imagenet_epochs_20':
        'https://drive.usercontent.google.com/download?id=1iK3t20-GS_Vs-_Q3ZexSiCfjGJ3IaPC2&export=download&authuser=0&confirm=t&uuid=d8f548aa-ccd6-4f0a-be49-356e9ee2e243&at=APZUnTXb7Ss81nGKgrixYS0binTs%3A1718048598551'
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
    'B-6_Wi-1024_res_64_cifar100_epochs_20':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-6_Wi-1024_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=11zFGFiKKxxrZOGk5oyk3AzBDnIY7KN3s&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar100_epochs_20':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-12_Wi-512_res_64_imagenet_epochs_50':
        'https://drive.google.com/uc?id=14GKtQ1iYwOqYpy4RcrWz2Ue3AGG7eGLz&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar10_epochs_20':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar100_epochs_40':
        'https://drive.google.com/uc?id=1Fjf4RA_yUXHgHncb9GIlf9zBNAJ-8giv&export=download&confirm=t&uuid',
    'B-6_Wi-512_res_64_cifar100_epochs_20':
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
