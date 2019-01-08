import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/* Starter Code @Author Kevin Quinn
 * Full code @Author Rui Ma
 * 
 * TextAssociator represents a collection of associations between words.
 * See write-up for implementation details and hints
 * 
 */
public class TextAssociator {
	private WordInfoSeparateChain[] table;
	private int size;
	private final int[] PRIMES = {11, 23, 47, 97, 199, 409, 821, 1669, 3389, 
			6827, 13649, 27299, 54601, 109201, 218401, 436801, 873617, 1747237,
			3494467, 6988931, 13977857};
	private int lengthIndex;    // index of the table length array
	
	/* INNER CLASS
	 * Represents a separate chain in your implementation of your hashing
	 * A WordInfoSeparateChain is a list of WordInfo objects that have all
	 * been hashed to the same index of the TextAssociator
	 */
	private class WordInfoSeparateChain {
		private List<WordInfo> chain;
		
		/* Creates an empty WordInfoSeparateChain without any WordInfo
		 */
		public WordInfoSeparateChain() {
			this.chain = new ArrayList<WordInfo>();
		}
		
		/* Adds a WordInfo object to the SeparateCahin
		 * Returns true if the WordInfo was successfully added, false otherwise
		 */
		public boolean add(WordInfo wi) {
			if (!chain.contains(wi)) {
				chain.add(wi);
				return true;
			}
			return false;	
		}
		
		/* Removes the given WordInfo object from the separate chain
		 * Returns true if the WordInfo was successfully removed, false otherwise
		 */
		public boolean remove(WordInfo wi) {
			if (chain.contains(wi)) {
				chain.remove(wi);
				return true;
			} 
			return false;
		}
		
		// Returns the size of this separate chain
		public int size() {
			return chain.size();
		}
		
		// Returns the String representation of this separate chain
		public String toString() {
			return chain.toString();
		}
		
		// Returns the list of WordInfo objects in this chain
		public List<WordInfo> getElements() {
			return chain;
		}
	}
	
	
	/* Creates a new TextAssociator without any associations 
	 */
	public TextAssociator() {
		size = 0;
		lengthIndex = 0;
		table = new WordInfoSeparateChain[PRIMES[lengthIndex]];
	}
	
	
	/* Adds a word with no associations to the TextAssociator 
	 * Returns False if this word is already contained in your TextAssociator
	 * If the load factor is too large, the table will resize
	 * Returns True if this word is successfully added
	 */
	public boolean addNewWord(String word) {
		if ((double)(size/table.length) > 1.0) {
			resize();
		}
		int index = hashIndex(word, table.length);
		if (table[index] == null) {
			table[index] = new WordInfoSeparateChain();
		} else {
			for (WordInfo words: table[index].getElements()) {
				if (words.getWord().equalsIgnoreCase(word)) {
					return false;
				}
			}	
		}
		table[index].add(new WordInfo(word));
		size++;
		return true;
	}
	
	
	/* Resizes the table
	 */
	private void resize() {
		lengthIndex++;
		WordInfoSeparateChain[] temp = new 
				WordInfoSeparateChain[PRIMES[lengthIndex]];
		for (WordInfoSeparateChain slots: table) {
			if (slots != null) {
				for (WordInfo words: slots.getElements()) {
					
					// Give words new indices
					int newIndex = hashIndex(words.getWord(), temp.length);      
					if (temp[newIndex] == null) {
						temp[newIndex] = new WordInfoSeparateChain();
					}
					temp[newIndex].add(words);
				}
			}
		}
		
		// Update the table
		table = temp;   
	}
	
	
	/* Adds an association between the given words. 
	 * Returns true if association correctly added, returns false if first 
	 * parameter does not already exist in the TextAssociator or if the 
	 * association between the two words already exists
	 */
	public boolean addAssociation(String word, String association) {
		int index = hashIndex(word, table.length);
		if (table[index] != null) {
			for (WordInfo words: table[index].getElements()) {
				if (words.getWord().equalsIgnoreCase(word) && 
						!words.getAssociations().contains(association)) {
					words.addAssociation(association);
					return true;
				}
			}	
		}	
		return false;
	}
	
	
	/* Removes the given word from the TextAssociator, returns false if word 
	 * was not contained, returns true if the word was successfully removed.
	 * Note that only a source word can be removed by this method, not an 
	 * association.
	 */
	public boolean remove(String word) {
		int index = hashIndex(word, table.length);
		if (table[index] != null) {
			for (WordInfo words: table[index].getElements()) {
				if (words.getWord().equalsIgnoreCase(word)) {
					table[index].remove(words);
					size--;
					return true;
				}
			}	
		}	
		return false;
	}
	
	
	/* Returns a set of all the words associated with the given String  
	 * Returns null if the given String does not exist in the TextAssociator
	 */
	public Set<String> getAssociations(String word) {
		int index = hashIndex(word, table.length);
		if (table[index] != null) {
			for (WordInfo words: table[index].getElements()) {
				if (words.getWord().equalsIgnoreCase(word)) {
					return words.getAssociations();
				}
			}	
		}
		return null;
	}
	
	
	/* Computes and returns the hash index for a given word and table length
	 */
	private int hashIndex(String word, int tableLength) {
		return Math.abs(word.hashCode() % tableLength);
	}
	
	
	/* Prints the current associations between words being stored
	 * to System.out
	 */
	public void prettyPrint() {
		System.out.println("Current number of elements : " + size);
		System.out.println("Current table size: " + table.length);
		
		//Walk through every possible index in the table
		for (int i = 0; i < table.length; i++) {
			if (table[i] != null) {
				WordInfoSeparateChain bucket = table[i];
				
				//For each separate chain, grab each individual WordInfo
				for (WordInfo curr : bucket.getElements()) {
					System.out.println("\tin table index, " + i + ": " + curr);
				}
			}
		}
		System.out.println();
	}
}
