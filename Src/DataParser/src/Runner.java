import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.FileWriter;
import java.io.IOException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.HashMap;

public class Runner {
    public static void main(String[] args) {
        // Replace with absolute project path
        final String PATH_TO_PROJECT = "/mnt/c/Data_Loc/OneDrive/Chase/IndepProjects/Nextdoor/Public";
        final String PATH_TO_CONFIG = PATH_TO_PROJECT.concat("/config.properties");
        final String PATH_TO_OUTPUT = PATH_TO_CONFIG.concat("/Src/DataParser/src/output_1.json");

        try {
            File configFile = new File(PATH_TO_CONFIG);
            User user = new User(new FileInputStream(configFile));
            NextdoorParser parser = new NextdoorParser('f', null, user);

            parser.login();
            parser.loadFeed();
            parser.scrollToEnd(3, 10);
            parser.readData();

            HashMap<Object, Object> data = parser.getData();

            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            String gsonData = gson.toJson(data);
            System.out.println(gsonData);

            FileWriter outputFile = new FileWriter(PATH_TO_OUTPUT);
            outputFile.write(gsonData);
            outputFile.close();
        
        } catch (FileNotFoundException e) {
            System.out.println("Config file not found!");
            System.exit(-1);
        } catch (IOException e) {
            System.out.println("Could not render results to output file!");
            System.exit(-1);
        }
    }
    
}
