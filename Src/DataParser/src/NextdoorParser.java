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


public class NextdoorParser extends Parser {
    private static final String LOGIN_URL = "https://nextdoor.com/login/";
    private static final String FEED_URL = "https://nextdoor.com/news_feed/?ordering=recent_activity";
    private User user;

    public NextdoorParser(char browser, String level, User user) {
        super(browser, level);
        this.user = user;
    }

    public boolean login() {
        navigateToURL(LOGIN_URL, 20);

        // Find username web element and input our username 
        WebElement username = driver.findElement(By.xpath("//input[@id='id_email']"));
        if (username == null) {
            super.printMsg("Cannot find username elem. Page layout changed?", "Fatal");
        }
        username.sendKeys(user.getUsername());

        // Find password web element and input our password 
        WebElement password = driver.findElement(By.xpath("//input[@id='id_password']"));
        if (password == null) {
            super.printMsg("Cannot find password elem. Page layout changed?", "Fatal");
        }
        password.sendKeys(user.getPassword());

        // Find the "sign in" web element and click the button 
        WebElement signIn = driver.findElement(By.xpath("//button[@id='signin_button']"));
        if (signIn == null) {
            super.printMsg("Cannot find signin elem. Page layout changed?", "Fatal");
        }
        signIn.click();

        // Once logged in, order posts by recency
        try {
            Thread.sleep(20000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        return true;
    }

    public void loadFeed() {
        navigateToURL(FEED_URL, 10);
    }

    public boolean clickClose(int sleep) {
        boolean success = true;
        try {
            WebElement parent = driver.findElement(By.xpath("//div[@aria-label='close'][@role='button']"));
            WebElement child = parent.findElement(By.xpath("./child::*"));
            child.click();
        } catch (org.openqa.selenium.NoSuchElementException e) {
            success = false;
        }

        try {
            Thread.sleep(20000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return success;

    }

    @Override
    public boolean readData() {
        List<WebElement> posts = driver.findElements(By.cssSelector(".css-2mived.css-1uyh6bs"));
        for (WebElement post: posts) {
            String postID = getID(post);

            if (postID == null || data.containsKey(postID)) {
                continue;
            }

            //System.out.println("ID is ".concat(postID)); 
            HashMap<String, Object> postData = readPost(post);
            data.put(postID, postData);
        }
        
        return true;
    }

    public HashMap<String, Object> readPost(WebElement post) {
        HashMap<String, Object> postData = new HashMap<String, Object>();

        postData.put("Title", getTitle(post));
        postData.put("Text", getText(post));
        postData.put("Author", getAuthor(post));
        postData.put("Age", getAge(post));
        postData.put("Location", getLocation(post));
        postData.put("NumComments", getNumComments(post));
        postData.put("NumReactions", getNumReactions(post));

        return postData;
    }

    private String getID(WebElement post) {
        String id = post.getAttribute("id");
        if (!id.isEmpty()) {
            return id.substring(2);
        }
        return null;
    }

    private String getTitle(WebElement post) {
        return post.getAttribute("aria-label"); 
    }

    private String getText(WebElement post) {
        try {
            WebElement container = post.findElement(By.xpath(".//p[@class='content-body']"));
            WebElement element = container.findElement(By.xpath(".//span[@class='Linkify']"));
            return element.getText();
        } catch (org.openqa.selenium.NoSuchElementException | NullPointerException e) {
            return "";
        }
    }

    private String getAuthor(WebElement post) {
        WebElement element = post.findElement(By.xpath(".//a[@class='_3I7vNNNM E7NPJ3WK']"));
        return element.getText();
    }

    private String getLocation(WebElement post) {
        WebElement metadata = post.findElement(By.xpath(".//span[@data-testid='author-children-test']"));
        List<WebElement> children = metadata.findElements(By.className("post-byline-redesign"));

        if (children != null) {
            return children.get(0).getText();
        }
        
        return null;
    }

    private String getAge(WebElement post) {
        WebElement metadata = post.findElement(By.xpath(".//span[@data-testid='author-children-test']"));
        List<WebElement> children = metadata.findElements(By.className("post-byline-redesign"));
        String age = children.get(1).getText();

        if (age != null) {
            return age;
        }
        return "";
    }

    private int getNumReactions(WebElement post) {
        try {
            WebElement parent = post.findElement(By.xpath(".//div[@data-testid='reaction-and-comment-counts']"));
            WebElement element = parent.findElement(By.xpath(".//div[@data-testid='count-text']"));
            return Integer.parseInt(element.getText());
        } catch (org.openqa.selenium.NoSuchElementException | NullPointerException e) {
            return 0;
        }
    }

    private int getNumComments(WebElement post) {
        try {
            WebElement parent = post.findElement(By.cssSelector(".css-1xr34d2"));
            WebElement element = parent.findElement(By.cssSelector(".css-imog9u.css-11p3kb3"));
            String count = element.getText().split(" ")[0];
            return Integer.parseInt(count);
        } catch (org.openqa.selenium.NoSuchElementException | java.lang.NumberFormatException e) {
            return 0;
        }
    }

    public User getUser() {
        return user;
    }

}

