# playwright_mid_decay.py
# Playwright script for MID decay analysis - downloads daily reports for last 4 months

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from playwright.sync_api import Playwright, sync_playwright
import os

def format_date(date):
    """Format date as MM/DD/YYYY for the web interface (month/day/year)"""
    return date.strftime("%m/%d/%Y")

def run_for_date(playwright: Playwright, date_str: str, loop_count: int) -> None:
    """Run Playwright automation for a specific date and save with structured filename"""
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    try:
        # Login to the application
        page.goto("https://goldie.vrio.app/auth/login")
        page.get_by_placeholder("email").click()
        page.get_by_placeholder("email").fill("team123@team123proton.com")
        page.get_by_placeholder("password").click()
        page.get_by_placeholder("password").fill("j8R@C#@!@35QE7^Tn8")
        page.get_by_role("button", name="Login").click()

        # Go directly to the report page
        page.goto("https://goldie.vrio.app/report/run/109/14", wait_until="networkidle")
        
        # Set date range for single day (same date on both ends)
        date_range = f"{date_str} - {date_str}"
        print(f"Setting date range to: {date_range}")
        
        # Set date range using the working approach with human-like typing
        page.locator("#rb_date_range").click()
        page.locator("#rb_date_range").click()
        page.locator("#rb_date_range").fill("")
        page.locator("#rb_date_range").click()
        page.locator("#rb_date_range").type(date_range, delay=50)
        page.get_by_role("button", name="Apply").click()
        
        # Add dimensions
        # Add Item dimension
        page.get_by_role("link", name="Add Dimension").click()
        page.get_by_role("textbox", name="Select Next Dimension").click()
        page.get_by_role("searchbox").fill("item")
        page.get_by_role("option", name="Item", exact=True).click()
        
        # Add Card Type dimension
        page.get_by_role("link", name="Add Dimension").click()
        page.get_by_role("textbox", name="Select Next Dimension").click()
        page.get_by_role("searchbox").fill("card")
        page.get_by_role("option", name="Card Type").click()
        
        # Add Transaction Day Of Week dimension
        page.get_by_role("link", name="Add Dimension").click()
        page.get_by_role("textbox", name="Select Next Dimension").click()
        page.get_by_role("option", name="Transaction Day Of Week").click()
        
        # Add Card Bin Number dimension
        page.get_by_role("link", name="Add Dimension").click()
        page.get_by_role("textbox", name="Select Next Dimension").click()
        page.get_by_role("searchbox").fill("bin")
        page.get_by_role("option", name="Card Bin Number", exact=True).click()
        
        # Add Ship State dimension
        page.get_by_role("link", name="Add Dimension").click()
        page.get_by_role("textbox", name="Select Next Dimension").click()
        page.get_by_role("option", name="Ship State").click()
        
        # Export the report
        page.get_by_role("link", name="Add Dimension").click()
        page.get_by_role("textbox", name="Select Next Dimension").click()
        page.get_by_role("button", name="More Options ").click()
        
        with page.expect_download() as download_info:
            page.get_by_role("link", name="Export Report").click()
        download = download_info.value
        
        # Save the downloaded file with structured name
        filename = f"db_decay_{loop_count}.csv"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        download.save_as(filepath)
        print(f"Downloaded and saved: {filename} for date {date_str}")
        
    except Exception as e:
        print(f"Error processing date {date_str}: {str(e)}")
    
    finally:
        context.close()
        browser.close()

def main():
    """Main function to loop through dates and download reports"""
    # Get current date
    today = datetime.now()
    
    # Calculate start date (4 months ago)
    start_date = today - relativedelta(months=4)
    
    print(f"Starting download from {format_date(start_date)} to {format_date(today)}")
    
    # Loop through each day from start date to today
    current_date = start_date
    loop_count = 1
    
    with sync_playwright() as playwright:
        while current_date <= today:
            formatted_date = format_date(current_date)
            print(f"Processing day {loop_count}: {formatted_date}")
            
            # Run Playwright automation for this date
            run_for_date(playwright, formatted_date, loop_count)
            
            # Move to next day and increment counter
            current_date += timedelta(days=1)
            loop_count += 1
    
    print(f"Completed! Downloaded {loop_count - 1} daily reports.")

if __name__ == "__main__":
    main()


