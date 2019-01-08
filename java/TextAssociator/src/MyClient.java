import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;


/* Starter code @Author: Kevin Quinn
 * Full code @Author: Rui Ma
 * 
 * MyClient takes a text file of famous tourist attractions in each state in
 * the US as specified by ATTRACTION_FILE and input from standard.in and
 * outputs a randomly selected attraction in the state that matches the user's
 * input.
 * 
 * Users can enter one or multiple states separated by "," each time.
 * 
 * This client program is dependent on TextAssociator.
 */
public class MyClient {
	
	//Path to desired attractions file to read
	public final static String ATTRACTION_FILE = "attractions.txt";
	
	public static void main(String[] args) throws IOException {
		File file = new File(ATTRACTION_FILE);
		
		//Create new empty TextAssociator
		TextAssociator sc = new TextAssociator();
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String text = null;
		
		while ((text = reader.readLine()) != null) {
			String[] words = text.split(",");
			String currWord = words[0].trim();
			sc.addNewWord(currWord);
			
			for (int i = 1; i < words.length; i++) {
				sc.addAssociation(currWord, words[i].trim());
			}
		}
		Scanner scan= new Scanner(System.in);
		String inputString = "";
		Random rand = new Random();
		System.out.println("Random famous attractions by states in the US:");
		while (true) {
			System.out.println();
			System.out.print("Please enter the full name of the state(s) (use "
				+ "\",\" to separate the states and enter \"exit\" to exit):");
			inputString = scan.nextLine();
			if (inputString.equals("exit")) {
				break;
			}
			String[] states  = inputString.split(",");
			String result = "";
			for (String state : states) {
				Set<String> attractions = 
						sc.getAssociations(state.trim().toLowerCase());
				if (attractions == null) {
					System.out.println("The state '" + state.trim() 
					+ "' cannot be found.");
				} else {
					System.out.println(state.trim() + ": " + 
				attractions.toArray()[rand.nextInt(attractions.size())]);
				}
			}
		}
		reader.close();
	}
}
