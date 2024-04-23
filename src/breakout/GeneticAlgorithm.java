package breakout;

import java.util.Random;
import java.util.Arrays;

public class GeneticAlgorithm {
   
	private static final int POPULATION_SIZE = 100;
	private static final int NUM_GENERATIONS = 100;  
	private static final int HIDDENLAYERS = 4; //alterar aqui o número de nós hidden layer que queremos- ir testando
	private static final double MUTATION_RATE = 0.05;
	private NeuralNetwork[] population = new NeuralNetwork[POPULATION_SIZE];
	private Random random;
	
	public GeneticAlgorithm() {
		generatePopulation();
		evolve();         
	}
	
	private void generatePopulation() {
		for (int i=0; i < population.length; i++) {
			population[i] = new NeuralNetwork(HIDDENLAYERS);   
		}
	}
	
	 public void evolve() {
	        for (int generation = 0; generation < NUM_GENERATIONS; generation++) {
	            evaluatePopulation();
	            NeuralNetwork[] parents = selectParents();
	            population = generateNewPopulation(parents);
	        }
	    }
	 
	 private void evaluatePopulation() {
	        for (NeuralNetwork network : population) {
	            double fitness = simulateGame(network);
	            network.setFitness(fitness);
	        }
	    }
	 
	 public double simulateGame(NeuralNetwork network) {
		  BreakoutBoard breakoutBoard = new BreakoutBoard(network, false, 1); //este 1 é o valor da seed
		  breakoutBoard.runSimulation();
		  double fitness = breakoutBoard.getFitness();
		  return fitness;
	 }
	 
	 private NeuralNetwork[] selectParents() {
	        // Sort population by fitness (descending order)
	        Arrays.sort(population, (n1, n2) -> Double.compare(n2.getFitness(), n1.getFitness()));

	        // Select top individuals as parents (elites)
	        NeuralNetwork[] parents = new NeuralNetwork[POPULATION_SIZE / 2];
	        System.arraycopy(population, 0, parents, 0, POPULATION_SIZE / 2);
	        return parents;
	    }
	 
	 private NeuralNetwork[] generateNewPopulation(NeuralNetwork[] parents) {
	        NeuralNetwork[] newPopulation = Arrays.copyOf(parents, POPULATION_SIZE); // Copy elite parents

	        int currentIndex = parents.length;
	        while (currentIndex < POPULATION_SIZE) {
	            NeuralNetwork parent1 = parents[random.nextInt(parents.length)];
	            NeuralNetwork parent2 = parents[random.nextInt(parents.length)];
	            mutate(parent1); //mutação
	            mutate(parent2); //mutação
	            NeuralNetwork offspring = crossover(parent1, parent2);   //crossover									
	            newPopulation[currentIndex++] = offspring;
	        }

	        return newPopulation;
	    }
	 
	 private NeuralNetwork crossover(NeuralNetwork parent1, NeuralNetwork parent2) {
		 	int inputDim = parent1.getInputDim();
		    int hiddenDim = parent1.getHiddenDim();
		    int outputDim = parent1.getOutputDim();

		    // Create a new neural network for the offspring
		    NeuralNetwork offspring = new NeuralNetwork(hiddenDim);

		    // Perform crossover for weights between parent1 and parent2
		    Random random = new Random();
		    for (int i = 0; i < inputDim; i++) {
		        for (int j = 0; j < hiddenDim; j++) {
		            // Randomly select weight from parent1 or parent2
		            if (random.nextBoolean()) {
		                offspring.setHiddenWeight(i, j, parent1.getHiddenWeight(i, j));
		            } else {
		                offspring.setHiddenWeight(i, j, parent2.getHiddenWeight(i, j));
		            }
		        }
		    }

		    for (int i = 0; i < hiddenDim; i++) {
		        for (int j = 0; j < outputDim; j++) {
		            // Randomly select weight from parent1 or parent2
		            if (random.nextBoolean()) {
		                offspring.setOutputWeight(i, j, parent1.getOutputWeight(i, j));
		            } else {
		                offspring.setOutputWeight(i, j, parent2.getOutputWeight(i, j));
		            }
		        }
		    }

		    // Return the offspring neural network after crossover
		    return offspring;
		}

	    private void mutate(NeuralNetwork network) {
	    	// Mutate hidden layer weights
	        for (int i = 0; i < network.getInputDim(); i++) {
	            for (int j = 0; j < network.getHiddenDim(); j++) {
	                if (random.nextDouble() < MUTATION_RATE) {
	                    // Add small random value to the weight
	                    double currentWeight = network.getHiddenWeight(i, j);
	                    double mutation = random.nextGaussian() * 0.1; // Small Gaussian mutation
	                    network.setHiddenWeight(i, j, currentWeight + mutation);
	                }
	            }
	        }
	        // Mutate output layer weights
	        for (int i = 0; i < network.getHiddenDim(); i++) {
	            for (int j = 0; j < network.getOutputDim(); j++) {
	                if (random.nextDouble() < MUTATION_RATE) {
	                    // Add small random value to the weight
	                    double currentWeight = network.getOutputWeight(i, j);
	                    double mutation = random.nextGaussian() * 0.1; // Small Gaussian mutation
	                    network.setOutputWeight(i, j, currentWeight + mutation);
	                }
	            }
	        }
	     // Mutate hidden layer biases
	        for (int i = 0; i < network.getHiddenDim(); i++) {
	            if (random.nextDouble() < MUTATION_RATE) {
	                // Add small random value to the bias
	                double currentBias = network.getHiddenBias(i);
	                double mutation = random.nextGaussian() * 0.1; // Small Gaussian mutation
	                network.setHiddenBias(i, currentBias + mutation);
	            }
	        }
	     // Mutate output layer biases
	        for (int i = 0; i < network.getOutputDim(); i++) {
	            if (random.nextDouble() < MUTATION_RATE) {
	                // Add small random value to the bias
	                double currentBias = network.getOutputBias(i);
	                double mutation = random.nextGaussian() * 0.1; // Small Gaussian mutation
	                network.setOutputBias(i, currentBias + mutation);
	            }
	        }
	    }
}
	 