import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

// Name: SUWIMOL KAMLANGJAI 
// Username: kamlangjai_s

// Bayesian Tomatoes:
// Doing some Naive Bayes and Markov Models to do basic sentiment analysis.
//
// Input from train.tsv.zip at 
// https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews
//
// itself gathered from Rotten Tomatoes.
//
// Format is PhraseID[unused]   SentenceID  Sentence[tokenized] Sentiment
//
// We'll only use the first line for each SentenceID, since the others are
// micro-analyzed phrases that would just mess up our counts.
//
// Sentiment is on a 5-point scale:
// 0 - negative
// 1 - somewhat negative
// 2 - neutral
// 3 - somewhat positive
// 4 - positive
//
// For each kind of model, we'll build one model per sentiment category.
// Following Bayesian logic, base rates matter for each category; if critics
// are often negative, that should be a good guess in the absence of other
// information.
//
// To play well with HackerRank, input is assumed to be the train.tsv
// format of training data until we encounter a line that starts with "---".
// All remaining lines, which should be just space-delimited words/tokens
// in a sentence, are assumed to be test data.
// Output is the following on four lines for each line of test data:
//
// Naive Bayes classification (0-4)
// Naive Bayes most likely class's log probability (with default double digits/precision)
// Markov Model classification (0-4)
// Markov Model most likely class's log probability

public class BayesianTomatoes {

    public static final int CLASSES = 5;
    // Assume sentence numbering starts with this number in the file
    public static final int FIRST_SENTENCE_NUM = 1;
    // Probability of either a unigram or bigram that hasn't been seen
    public static final double OUT_OF_VOCAB_PROB = 0.0000000001;

    // Sorry about the "global"ish variables here, but it's going to
    // make all the other signatures rather cleaner

    // Word counts for each sentiment label
    public static ArrayList<HashMap<String, Integer>> wordCounts;
    // Bigram counts for each sentiment label, with key a single string
    // separating the words with a space
    public static ArrayList<HashMap<String, Integer>> bigramCounts;
    // Overall sentence sentiment counts for taking the prior into account
    // (one is incremented once per sentence)

    // A subtle point:  if a word is at the end of the sentence, it's not
    // the beginning of any bigram.  So we need to keep separate track of
    // the number of times a word starts any bigram (ie is not the last word)
    public static ArrayList<HashMap<String, Integer>> bigramDenoms;

    public static int[] sentimentCounts;
    // total number of words in a class (sentiment)
    public static int[] totalWords;
    // total number of bigrams in a class
    public static int[] totalBigrams;

    public static class Classification {
        public int rating;       // the maximum likelihood classification
        public double logProb;   // the log likelihood of that classification

        public Classification(int c, double lp) {
            rating = c;
            logProb = lp;
        }

        public String toString() {
            return String.format("%d\n%.5f\n", rating, logProb);
        }
    }

    public static void main(String[] args) {
        Scanner myScanner = new Scanner(System.in);   
        getModels(myScanner);
        classifySentences(myScanner);
    }

    public static void getModels(Scanner sc) {
        int nextFresh = FIRST_SENTENCE_NUM;
        initializeStructures();
        while(sc.hasNextLine()) {
            String line = sc.nextLine();
            if (line.startsWith("---")) {
                return;
            }
            String[] fields = line.split("\t");
            try {
                Integer sentenceNum = Integer.parseInt(fields[1]);
                if (sentenceNum != nextFresh) {
                    continue;
                }
                nextFresh++;
                Integer sentiment = Integer.parseInt(fields[3]);
                sentimentCounts[sentiment]++;
                updateWordCounts(fields[2], wordCounts.get(sentiment),
                                 bigramCounts.get(sentiment), 
                                 bigramDenoms.get(sentiment),
                                 sentiment);
            } catch (Exception e) {
                // We probably just read the header of the file.
                // Or some other junk.  Ignore.
            }
        }
    }

    // Initialize the global count data structures
    public static void initializeStructures() {
        sentimentCounts = new int[CLASSES];
        totalWords = new int[CLASSES];
        totalBigrams = new int[CLASSES];
        wordCounts = new ArrayList<HashMap<String, Integer>>();
        bigramCounts = new ArrayList<HashMap<String, Integer>>();
        bigramDenoms = new ArrayList<HashMap<String, Integer>>();
        for (int i = 0; i < CLASSES; i++) {
            wordCounts.add(new HashMap<String, Integer>());
            bigramCounts.add(new HashMap<String, Integer>());
            bigramDenoms.add(new HashMap<String, Integer>());
        }
    }

    // updateWordCounts:  assume space-delimited words/tokens
    // notice that we are shadowing the globals with sentiment-specific
    // hashmaps
    public static void updateWordCounts(String sentence, 
                                        HashMap<String, Integer> wordCounts, 
                                        HashMap<String, Integer> bigramCounts, 
                                        HashMap<String, Integer> bigramDenoms,
                                        int sentiment) {
        String[] tokenized = sentence.split(" ");
        for (int i = 0; i < tokenized.length; i++) {
            totalWords[sentiment]++;
            String standardized = tokenized[i].toLowerCase();
            if (wordCounts.containsKey(standardized)) {
                wordCounts.put(standardized, wordCounts.get(standardized)+1);
            } else {
                wordCounts.put(standardized, 1);
            }
            if (i > 0) {
                String bigram = (tokenized[i-1] + " " + tokenized[i]).toLowerCase();
                if (bigramCounts.containsKey(bigram)) {
                    bigramCounts.put(bigram, bigramCounts.get(bigram) + 1);
                } else {
                    bigramCounts.put(bigram, 1);
                }

                String standardizedPrev = tokenized[i-1].toLowerCase();
                if (bigramDenoms.containsKey(standardizedPrev)) {
                    bigramDenoms.put(standardizedPrev, bigramDenoms.get(standardizedPrev) + 1);
                } else {
                    bigramDenoms.put(standardizedPrev, 1);
                }
                totalBigrams[sentiment]++;
            }
        }
    }

    // Assume test data consists of just space-delimited words in sentence
    public static void classifySentences(Scanner sc) {
        while(sc.hasNextLine()) {
            String line = sc.nextLine();
            Classification nbClass = naiveBayesClassify(line);
            Classification mmClass = markovModelClassify(line);
           
            System.out.print(nbClass.toString() + mmClass.toString());
        }
      
    }

    // Classify a new sentence using the data and a Naive Bayes model.
    // Assume every token in the sentence is space-delimited, as the input
    // was.
    public static Classification naiveBayesClassify(String sentence) {
        // TODO
      int allSentimentCounts = sentimentCounts[0] + sentimentCounts[1] + sentimentCounts[2] + sentimentCounts[3] + sentimentCounts[4];
      
      // initialize the prior for each sentiment
      double[] priorSent = {0, 1, 2, 3, 4};
        
        // initializing the prior for each sentiment
        for (int i = 0; i < CLASSES; i++) {
            priorSent[i] = Math.log(sentimentCounts[i]) - Math.log(allSentimentCounts);
        }
      
      // split the given sentence and convert all words to lower case
      String[] words = sentence.split(" ");
      for (int i = 0; i < words.length; i++) {
        words[i] = words[i].toLowerCase();
      }
      
      // then find prob of word for a particular sentence in a particular sentiment, Pr(word | sent 0),
      // Pr(word | sent 1), ... , Pr(word | sent 4)
      // then update the initial prob
      
      // initialize the result probability
      // maxProb is the log prob that we're returning
      double maxProb = Double.NEGATIVE_INFINITY;
      int rating = -1; // value is initially -1
      for (int i = 0; i < CLASSES; i++) {
        for (int j = 0; j < words.length; j++) {
          double numerator = 0;
            double denominator = 0;

            // calculate prob of a word in the sentence with a particular sentiment
            if (wordCounts.get(i).containsKey(words[j])) {
              numerator = wordCounts.get(i).get(words[j]);
              denominator = totalWords[i];
              
              // find the prob
              priorSent[i] = Math.log(numerator) - Math.log(denominator) + priorSent[i];
            } else {
              // if word hasn't been seen:
              priorSent[i] = priorSent[i] + Math.log(OUT_OF_VOCAB_PROB);
            }
        }
        // finally, update the current maximum probability and rating
          if (priorSent[i] > maxProb) {
                maxProb = priorSent[i];
            rating = i;
          }
      }

        return new Classification(rating, maxProb);
    }

    // Like naiveBayesClassify, but each word is conditionally dependent
    // on the preceding word.
    public static Classification markovModelClassify(String sentence) {
        // TODO
      // Part 1: same as Naive Bayes'
      int allSentimentCounts = sentimentCounts[0] + sentimentCounts[1] + sentimentCounts[2] + 
          sentimentCounts[3] + sentimentCounts[4];
      
      // initialize the prior for each sentiment
      double[] priorSent = {0, 1, 2, 3, 4};
        
        // initializing the prior for each sentiment
        for (int i = 0; i < CLASSES; i++) {
            priorSent[i] = Math.log(sentimentCounts[i]) - Math.log(allSentimentCounts);
        }
      
        // split the given sentence and convert all words to lower case
        String[] words = sentence.split(" ");
      for (int i = 0; i < words.length; i++) {
        words[i] = words[i].toLowerCase();
      }
        
      double maxProb = Double.NEGATIVE_INFINITY;
      int rating = -1; // value is initially -1
    
      for (int i = 0; i < CLASSES; i++) {
        for (int j = 0; j < words.length; j++) { // starting from second word in the chain
          double numerator = 0;
            double denominator = 0;
            
            // find prob for the first word in word chain (unigram / same as Naive Bayes)
            if (j == 0) {
              // calculate prob of a word in the sentence with a particular sentiment
                if (wordCounts.get(i).containsKey(words[j])) {
                  numerator = wordCounts.get(i).get(words[j]);
                  denominator = totalWords[i];
                  // find the prob
                  priorSent[i] = Math.log(numerator) - Math.log(denominator) + priorSent[i];
                } else {
                  // if word hasn't been seen:
                  priorSent[i] = priorSent[i] + Math.log(OUT_OF_VOCAB_PROB);
                }
            } else { // find prob for the rest in word chain (bigram)
              // calculate prob of a word in the sentence with a particular sentiment
                if (bigramCounts.get(i).containsKey(words[j-1] + " " + words[j])) {
                  numerator = bigramCounts.get(i).get(words[j-1] + " " + words[j]);
                  
                  if (bigramDenoms.get(i).containsKey(words[j-1])) {
                    denominator = bigramDenoms.get(i).get(words[j-1]);
                    } else {
                      denominator = totalBigrams[i];
                    }
                  
                  // find the prob
                  priorSent[i] = Math.log(numerator) - Math.log(denominator) + priorSent[i];
                } else {
                  // if word hasn't been seen:
                  priorSent[i] = priorSent[i] + Math.log(OUT_OF_VOCAB_PROB);
                }
            }
        }
        
        // finally, update the current maximum probability and rating

          if (priorSent[i] > maxProb) {
                maxProb = priorSent[i];
            rating = i;
          }
      }
      
        return new Classification(rating, maxProb);
    }
}

