import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;

/*
 * Inheriter of Parser class to manage runs collecting data for training models. 
 */
public class HistoricalParser extends NextdoorParser {
    private static final String FEED_URL = "https://nextdoor.com/news_feed/";

    public HistoricalParser(char browser, String level, User user) {
        super(browser, level, user);
    }

    @Override
    public void loadFeed() {
        navigateToURL(FEED_URL, 10);
    }
    
    @Override
    public HashMap<String, Object> readPost(WebElement post) {
        HashMap<String, Object> postData = super.readPost(post);
        boolean hasInteracted = hasInteracted(post, super.getUser().getName(), postData);

        postData.put("Interacted", hasInteracted);
        return postData;
    }

    private boolean hasInteracted(WebElement post, String name, HashMap<String, Object> postData) {
        /* Check if a user interacted with a particular post.
         * @param post: Post element
         * @param name: Name of user
         * @param postData: HashMap of data already parsed from post
         */
        return (hasReacted(post, name, postData) || hasCommented(post, name, postData));
    }

    private boolean hasReacted(WebElement post, String name, HashMap<String, Object> postData) {
        /* Check if a user reacted to a particular post (i.e. liked, cried, etc.). */

        // Skip if post has no reactions
        if (postData.get("NumReactions").equals(0)) {
            return false;
        }

        JavascriptExecutor js = (JavascriptExecutor) driver;

        // Click button to view more reactions
        // Note: Selenium click() function has bug for large viewports. We must execute JavaScript code to work around this
        WebElement reactionsButton = post.findElement(By.xpath(".//button[@data-testid='view-reactors-button']"));
        js.executeScript("arguments[0].click();", reactionsButton);
        sleep(5);

        WebElement parent = driver.findElement(By.cssSelector(".css-z0jtb0.css-1uyh6bs"));
        List<WebElement> elements = parent.findElements(By.xpath(".//h2[@class='css-393j1z']"));

        // Iterate through list of reactors and returns if target user was a reactor
        boolean foundReaction = false;
        for (WebElement element: elements) {
            if (name.equals(element.getText())) {
                foundReaction = true;
                break;
            }
        }
        
        // Close the pop-up dialog when done
        clickClose(5);
        return foundReaction;

    }

    private boolean hasCommented(WebElement post, String name, HashMap<String, Object> postData) {
        /* Check if a user left a comment on a particular post. */

        // Skip if post has no comments
        if (postData.get("NumComments").equals(0)) {
            return false;
        }


        JavascriptExecutor js = (JavascriptExecutor) driver;

        // Click button to load comments
        // Note: Selenium click() function has bug for large viewports. We must execute JavaScript code to work around this
        WebElement commentsButton = post.findElement(By.xpath(".//span[@data-testid='reply-button-label']"));
        js.executeScript("arguments[0].click();", commentsButton);
        sleep(5);

        // Load more comments if necessary
        try {
            WebElement viewMoreButton = post.findElement(By.cssSelector(".see-previous-comments-button-paged"));
            js.executeScript("arguments[0].click();", viewMoreButton);
        } catch (org.openqa.selenium.NoSuchElementException e) {
            // If there are no more comments to load, do nothing
        }
        sleep(5);

        // Iterate through list of commenters and returns if target user left a comment
        List<WebElement> elements = post.findElements(By.xpath(".//a[@class='comment-detail-author-name']"));
        for (WebElement element: elements) {
            if (name.equals(element.getText())) {
                return true;
            }
        }
        
        return false;
    }
}

