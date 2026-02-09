import requests
from bs4 import BeautifulSoup

URL = "https://careers.publicisgroupe.com/epsilon/jobs/134823?lang=en-us"
OUTPUT_FILE = "scraped_jobs.txt"

def get_job_details(link):
    """
    Fetches the detail page and extracts the job description.
    """
    try:
        response = requests.get(link)
        if response.status_code != 200:
            return "Failed to retrieve description"
        
        soup = BeautifulSoup(response.content, "html.parser")
        # On this specific site, the description is in a div with class 'content'
        # inside the div class 'box' usually, but 'content' class is unique enough here.
        content_div = soup.find("div", class_="content")
        if content_div:
            return content_div.get_text(separator="\n", strip=True)
        else:
            # Fallback if specific class not found
            return "Description not found."
    except Exception as e:
        return f"Error fetching description: {e}"

def scrape_jobs():
    print(f"Scraping {URL}...")
    response = requests.get(URL)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # The main container for results
    results = soup.find(id="ResultsContainer")
    if not results:
        print("Could not find results container.")
        return

    job_elements = results.find_all("div", class_="card-content")
    
    print(f"Found {len(job_elements)} job cards. Processing the first 5 as a demo...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # Header for the file
        f.write(f"Scraped Data from {URL}\n")
        f.write("=" * 50 + "\n\n")

        # Process only first 5 to be quick
        for job_elem in job_elements[:5]:
            title_elem = job_elem.find("h2", class_="title")
            company_elem = job_elem.find("h3", class_="company")
            location_elem = job_elem.find("p", class_="location")
            
            # Find the Apply link by iterating over 'a' tags or selecting specific class
            # The 'Apply' button is the second link usually, but searching by text is safer.
            links = job_elem.find_all("a")
            apply_link = None
            for link in links:
                if link.text.strip() == "Apply":
                    apply_link = link["href"]
                    break
            
            if title_elem and company_elem and location_elem and apply_link:
                title = title_elem.text.strip()
                company = company_elem.text.strip()
                location = location_elem.text.strip()
                
                print(f" -> Fetching details for: {title}")
                description = get_job_details(apply_link)
                
                # Write to file
                f.write(f"Title:       {title}\n")
                f.write(f"Company:     {company}\n")
                f.write(f"Location:    {location}\n")
                f.write(f"Description:\n{description}\n")
                f.write("-" * 50 + "\n")
            
    print(f"\nSuccess! Data saved to '{OUTPUT_FILE}'")

if __name__ == "__main__":
    scrape_jobs()
