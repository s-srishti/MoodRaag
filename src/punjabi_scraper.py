import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin
class PunjabiFolkSongScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.base_url = "https://www.lyricsmint.com/punjabi"
        self.songs_data = []
    def scrape_punjabi_songs(self, max_pages=2):
        for page in range(1, max_pages + 1):
            if page == 1:
                url = self.base_url
            else:
                url = f"{self.base_url}?page={page}"
            print(f"Fetching page {page}: {url}")
            try:
                response = requests.get(url, headers=self.headers, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                print(f"Page title: {soup.title.string if soup.title else 'No title'}")
                song_urls = []
                selectors_to_try = [
                    "div.container a[href*='/punjabi/']",
                    "a[href*='/punjabi/']",
                    ".block a",
                    "div.block a"
                ]
                for selector in selectors_to_try:
                    links = soup.select(selector)
                    if links:
                        print(f"Found {len(links)} links with selector: {selector}")
                        for link in links:
                            href = link.get('href')
                            if href:
                                print(f"  Link found: {href}") 
                                if (href.startswith('/') and
                                    not any(x in href.lower() for x in ['#', 'javascript:', 'mailto:', 'tel:']) and
                                    len(href.split('/')) >= 3 and  
                                    href not in song_urls):
                                    song_urls.append(href)
                        break
                print(f"Found {len(song_urls)} unique song URLs on page {page}")
                for i, song_url in enumerate(song_urls): 
                    if not song_url.startswith('http'):
                        full_url = "https://www.lyricsmint.com" + song_url
                    else:
                        full_url = song_url
                    print(f"Processing song {i+1}/{len(song_urls)}: {full_url}")
                    self.scrape_individual_song(full_url)
                    time.sleep(2)  
                print(f"Completed page {page}. Total songs collected: {len(self.songs_data)}")
                time.sleep(3)  
            except requests.RequestException as e:
                print(f"Network error on page {page}: {e}")
            except Exception as e:
                print(f"Unexpected error on page {page}: {e}")
    def scrape_individual_song(self, song_url):
        try:
            response = requests.get(song_url, headers=self.headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            title = self.extract_title(soup)
            metadata = self.extract_metadata(soup)
            lyrics = self.extract_lyrics(soup)
            if title and lyrics:
                song_data = {
                    'title': title,
                    'artist': metadata.get('singer', 'Unknown'),
                    'music': metadata.get('music', 'Unknown'),
                    'lyricist': metadata.get('lyricist', 'Unknown'),
                    'lyrics': lyrics,
                    'url': song_url,
                    'language': 'punjabi',
                }
                self.songs_data.append(song_data)
                print(f"Successfully scraped: {title}")
            else:
                print(f"Missing title or lyrics for: {song_url}")

        except requests.RequestException as e:
            print(f"Network error scraping {song_url}: {e}")
        except Exception as e:
            print(f"Error scraping {song_url}: {e}")
    def extract_title(self, soup):
        title_selectors = [
            "div.pt-4.pb-2 h2",
            "h1",
            "h2",
            ".title",
            "[class*='title']"
        ]
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                title = title_element.get_text(strip=True)
                title = title.replace("Lyrics", "").replace("lyrics", "").strip()
                if title:
                    return title
        if soup.title:
            title = soup.title.string.replace("Lyrics", "").replace("lyrics", "").strip()
            if title:
                return title
        return None
    def extract_metadata(self, soup):
        metadata = {}
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower()
                    value_cell = cells[1]
                    anchor = value_cell.find("a")
                    value = anchor.get_text(strip=True) if anchor else value_cell.get_text(strip=True)

                    if "singer" in key or "artist" in key:
                        metadata["singer"] = value
                    elif "music" in key or "composer" in key:
                        metadata["music"] = value
                    elif "lyricist" in key or "writer" in key:
                        metadata["lyricist"] = value
        return metadata
    def extract_lyrics(self, soup):
        lyrics_selectors = [
            "div.text-base.lg\\:text-lg .pb-2",
            "div[class*='text-base'] p",
            ".lyrics p",
            "[class*='lyrics'] p",
            "div p"
        ]
        for selector in lyrics_selectors:
            content_div = soup.select_one(selector.split()[0]) 
            if content_div:
                paragraphs = content_div.find_all("p")
                lyrics_lines = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if len(text) > 10 and not any(word in text.lower() for word in ['advertisement', 'ads', 'click', 'visit']):
                        lyrics_lines.append(text)

                if lyrics_lines:
                    return "\n".join(lyrics_lines)
        return None
    def save_data(self, filename='punjabi_lyricsmint_songs.json'):
        if not self.songs_data:
            print("No songs data to save!")
            return None
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.songs_data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.songs_data)} songs to {filename}")
        if self.songs_data:
            print("\n Summary:")
            print(f"Total songs: {len(self.songs_data)}")
            print("Sample titles:")
            for i, song in enumerate(self.songs_data):
                print(f"  {i+1}. {song['title']} - {song['artist']}")
        return filename
    def debug_page_structure(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"\nDebug info for: {url}")
            print(f"Page title: {soup.title.string if soup.title else 'None'}")
            print(f"Total links found: {len(soup.find_all('a'))}")
            print(f"Total divs found: {len(soup.find_all('div'))}")
            print("\nSample links found:")
            for i, link in enumerate(soup.find_all('a')[:15]):
                href = link.get('href', 'No href')
                text = link.get_text(strip=True)[:50]
                print(f"  {i+1}. {href} -> '{text}'")
            print("\nSample div classes:")
            for div in soup.find_all('div', class_=True)[:10]:
                classes = ' '.join(div.get('class', []))
                print(f"Div classes: {classes}")
        except Exception as e:
            print(f"Debug error: {e}")
if __name__ == "__main__":
    scraper = PunjabiFolkSongScraper()
    print("Debugging page structure...")
    scraper.debug_page_structure("https://www.lyricsmint.com/punjabi")
    print("\nStarting scraping...")
    scraper.scrape_punjabi_songs(max_pages=299) 
    if scraper.songs_data:
        scraper.save_data()
    else:
        print(" No songs were scraped. Check the page structure and selectors.")