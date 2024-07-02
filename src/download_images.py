import aiohttp
import asyncio
import flickrapi
import json
import os

from dotenv import load_dotenv
from tqdm import tqdm

# Load Flickr API keys from .env
load_dotenv(".env")
FLICKR_API_KEY = os.environ["FLICKR_API_KEY"]
FLICKR_API_SECRET = os.environ["FLICKR_API_SECRET"]

# Sierra Nevada bounding box coordinates 
MIN_LONGITUDE = "-3.488159"
MAX_LONGITUDE = "-2.739716"
MAX_LATITUDE = "37.179467"
MIN_LATITUDE = "36.956476"

# Other constants ...
SAVE_DIR = 'data/images'

async def download_image(session, semaphore, image_url, save_dir, image_name=None):
    async with semaphore:
        async with session.get(image_url) as response:
            if response.status == 200:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                if image_name is None:
                    image_name = os.path.basename(image_url)

                save_path = os.path.join(save_dir, image_name)

                with open(save_path, 'wb') as f:
                    f.write(await response.read())

                #print(f"Image successfully downloaded: {save_path}")
            else:
                print(f"Failed to download image. Status code: {response.status}")

async def download_images(
        image_urls: list[str], 
        save_dir: str, 
        max_concurrent_requests: int = 10, 
        image_names: list[str] = None):
    if image_names is not None:
        assert len(image_urls) == len(image_names), \
                "image_names and image_urls have different sizes"
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, image_url in enumerate(image_urls):
            tasks.append(download_image(session, semaphore, image_url, save_dir, str(image_names[i])))
        await asyncio.gather(*tasks)

async def download_flickr_images(
        coordinates: list[str], 
        save_dir: str, 
        **kwargs) -> int:
    """
        Download flickr images for a given coordinates.

        Args:
            - coordinates (list[str]): List of coordinates - minimum longitude, 
            minimum latitude, maximum longitude,maximum latitude
            - save_dir (str): Diretory to save images
            - **kwargs (dict): Rest of flicker API arguments for `photos.search` 
            method

        Returns:
            - int: Number of photos downloaded
    """

    flickr = flickrapi.FlickrAPI(
        FLICKR_API_KEY, 
        FLICKR_API_SECRET,
        format='json'
    )

    bbox = ",".join(coordinates)
    params = kwargs
    params['bbox'] = bbox
    params['page'] = 1
    params['per_page'] = 250

    photos = flickr.photos.search(
        **params
    )
    photos = json.loads(photos)

    total_pages = photos['photos']['pages']
    total_photos = photos['photos']['total']

    for page in tqdm(range(1, total_pages+1), 
            desc="Downloading {} images".format(total_photos)):

        params['page'] = page
        photos = flickr.photos.search(
            **params
        )
        photos = json.loads(photos)
        photos = photos['photos']['photo']

        if len(photos) > 0:
            page_urls = [photo.get('url_m') for photo in photos]
            range_start = params['per_page']*(page-1)
            range_end = range_start + len(page_urls)
            image_names = [i for i in range(range_start, range_end)]
            await download_images(page_urls, save_dir, image_names=image_names)

    print("successfully downloaded {} images into {}".format(
        total_photos, save_dir))
    return total_photos

if __name__ == "__main__":

    coordinates = [MIN_LONGITUDE, MIN_LATITUDE,
                   MAX_LONGITUDE, MAX_LATITUDE]

    asyncio.run(
        download_flickr_images(
            coordinates=coordinates,
            save_dir=SAVE_DIR,
            min_upload_date='2004-01-01 00:00:00',
            max_upload_date='2024-06-27 00:00:00',
            extras='url_m,date_upload'
        )
    )
