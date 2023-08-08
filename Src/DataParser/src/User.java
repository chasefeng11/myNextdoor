import java.util.Properties;
import java.io.FileInputStream;
import java.io.IOException;

/*
 * Class for storing user's Nextdoor account information.
 * Used for logging in, training models, etc.
 */
public class User {
    private String name;
    private String username;
    private String password;

    public User(String fullName, String username, String password) {
        /* Constructor to read values.
         * @param name: User's full name
         * @param username: User's email address as registered on Nextdoor
         * @param password: User's password for Nextdoor
        */
        this.name = fullName;
        this.username = username;
        this.password = password;
    }

    public User(FileInputStream input) {
        /* Constructor to read values.
         * @param input: FileInputStream for config file
        */
        Properties prop = new Properties();

        try {
            prop.load(input);

            this.name = prop.getProperty("Name");
            this.username = prop.getProperty("Username");
            this.password = prop.getProperty("Password");
        
        }   catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public String getName() { return this.name; }
    public String getUsername() { return this.username; }
    public String getPassword() { return this.password; }
}
