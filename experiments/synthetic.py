import torch
from diffusers import StableDiffusionPipeline
import cv2
import os
import tqdm

def run_synthetic_pipeline(args):

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    # downloaded models are big (~5gb), add "cache_dir" argument to change download location

    if args.device == "available":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    pipe = pipe.to(device)

    save_dir = args.synthetic_path
    os.makedirs(save_dir, exist_ok=True)

    # prompt = "a photo of a chameleon with same color and texture as tree background"
    # images = pipe(prompt).images
    # print(len(images))
    # image = images[0]
    # image.save("diffusion.png") 

    animals = [
        "pipefish",
        "Katydid",
        "Spider",
        "Grasshopper",
        "Cat",
        "Bird",
        "Toad",
        "Lizard",
        "Owl",
        "SeaHorse",
        "butterfly",
        "mantis",
        "Frog",
        "Caterpillar",
        "Cicada",
        "ScorpionFish",
        "Fish",
        "GhostPipefish",
        "Crab",
        "Moth",
        "Stickinsect",
        "Chameleon",
        "Snake",
        "Dog",
        "Heron",
        "Gecko",
        "Leopard",
        "Flounder",
        "Deer",
        "Octopus",
        "Dragonfly",
        "Mockingbird",
        "tiger",
        "lion",
        "cheetah",
        "leapard",
        "rat",
        "elephant",
        "zebra",
        "hippo",
        "giraffe",
        "duck",
        "monkey",
        "rabbit",
        "raccoon",
        "sheep",
        "wolf",
        "crocodile",
        "slug",
        "turtle",
    ]

    colors = [
        "green",
        "brown",
        "blue",
        "white",
        "gray",
    ]

    # prompts = [
    #     "a photo of a camouflaged butterfly",
    #     "a photo of a camouflaged chameleon",
    #     "a photo of a butterfly hiding in plain sight",
    #     "a photo of a chameleon hiding in plain sight",
    # ]
    # prompts += [
    #     "a photo of a camouflaged {}".format(animal) 
    #     for animal in animals
    # ]

    # prompts += [
    #     "a photo of an animal concealed in nature",
    #     "a photo of an animal concealed in a forest",
    #     "a photo of a butterfly concealed in a forest",
    #     "a photo of a butterfly hiding in a forest",
    # ]
    # prompts += [
    #     "a photo of a {} concealed in nature".format(animal) 
    #     for animal in animals
    # ]
    # prompts += [
    #     "a photo of a {} hiding in nature".format(animal) 
    #     for animal in animals
    # ]
    # prompts += [
    #     "a photo of a {} camouflaged in nature".format(animal) 
    #     for animal in animals
    # ]

    # prompts = [
    #     "a photo of a green colored {} in green forest with matching texture".format(animal) 
    #     for animal in animals
    # ]

    # reps = 1
    # print(len(prompts))
    # for i, p in enumerate(tqdm.tqdm(prompts)):
    #     for j in range(reps):
    #         images = pipe(p).images
    #         image = images[0]
    #         image.save(os.path.join(save_dir, "gen_{}_{}.png".format(i, j))) 

    reps = 20
    animal_color_reps = [
        (animal, c, r) 
        for animal in animals 
        for c in colors
        for r in range(reps)
    ]


    for animal, color, rep in tqdm.tqdm(animal_color_reps):
        p = "a photo of a {} colored {} in {} forest with matching texture".format(color, animal, color) 
        images = pipe(p).images
        image = images[0]
        image.save(os.path.join(save_dir, "gen_{}_{}_{}.png".format(animal, color, rep))) 