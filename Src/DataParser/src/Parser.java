import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.firefox.FirefoxDriver;
import org.openqa.selenium.JavascriptExecutor;

import java.util.HashMap;
import java.time.Duration;
import java.util.concurrent.TimeUnit;

public abstract class Parser {
    protected WebDriver driver;
    protected String warnLevel = "Warn";
    protected HashMap<Object, Object> data;

    public Parser(char browser, String level) {
        //seleniumClean(5);
        
        switch (browser) {
            case 'c':
                this.driver = new ChromeDriver();
                break;
            case 'f':
                this.driver = new FirefoxDriver();
                System.out.println("Driver created!");
                break;
            default:
                printMsg("Option not supported!", null);
                break;
        }

        this.data = new HashMap<>();
        this.warnLevel = level;
    }

    public void navigateToURL(String url, int sleep) {
        driver.get(url);
        try {
            Thread.sleep(sleep * 1000);
         } catch (InterruptedException e) {
            e.printStackTrace();
         }
    }

    private static void seleniumClean(int sleep) {
        int x = 0;
    }

    protected boolean printMsg(String msg, String warning) {
        System.out.print(msg);
        return true;
    }


    abstract public boolean readData();

    // Don't know if these need to be abstract
    public HashMap<Object, Object> getData() {
        return data;
    }

    public boolean printData() {
        return true;
    }

    public void scrollToEnd(int n, int sleep) {
        JavascriptExecutor js = (JavascriptExecutor) driver;
        for (int i = 0; i < n; i++) {
            js.executeScript("window.scrollTo(0, document.body.scrollHeight)");
            try {
                Thread.sleep(1000 * sleep);
             } catch (InterruptedException e) {
                e.printStackTrace();
             }
        }
    } 

    public void sleep(int seconds) {
        try {
            Thread.sleep(1000 * seconds);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

