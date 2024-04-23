package breakout;

import java.util.Random;

import utils.GameController;

public class NeuralNetwork implements GameController {
	//private static int inputDim;
	private static int INPUTDIM = 7;
	private static int OUTPUTDIM = 1;
	private int hiddenDim;
	//private int outputDim;
	private double[][] hiddenWeights;
	private double[] hiddenBiases;
	private double[][] outputWeights;
	private double[] outputBiases;
	private double fitness;
	
	public NeuralNetwork(int hiddenDim) {
		this.hiddenDim=hiddenDim;
		hiddenWeights = new double [INPUTDIM][hiddenDim];
		outputWeights= new double [hiddenDim][OUTPUTDIM];
		hiddenBiases = new double [hiddenDim];
		outputBiases = new double [OUTPUTDIM];
		
		initializeParameters();
    }
	
	 public void initializeParameters() {
	    	Random random = new Random();
			

			for(int i=0; i < INPUTDIM; i++) {
				for(int j=0; j < hiddenDim; j++) {
					hiddenWeights[i][j] = random.nextDouble() * 0.5;
				}
			}
			
			for(int i=0; i < hiddenDim; i++) {
				for(int j=0; j < OUTPUTDIM; j++) {
					outputWeights[i][j] = random.nextDouble() * 0.5;
				}
			}
			
			for(int i=0; i < hiddenDim; i++) {
				hiddenBiases[i] = random.nextDouble() * 0.5;
			}
			
			for(int i=0; i < OUTPUTDIM; i++) {
				outputBiases[i] = random.nextDouble() * 0.5;
			}
	    }
	 	
		public double sigmoid(double val) {
			return (1 / (1 + Math.exp(-val)));
		}
		
		public int func(double val) {
			if (val < 0.5) {
				return 1;
			}
			else {
				return 2;
			}
		}
		
		public int getInputDim() {
			return INPUTDIM;
		}
		
		public int getOutputDim() {
			return OUTPUTDIM;
		}
		
		public int getHiddenDim() {
			return hiddenDim;
		}
		public double getHiddenBias(int i) {
			return hiddenBiases[i];
		}
		
		public double getOutputBias(int i) {
			return outputBiases[i];
		}
		
		public double getHiddenWeight(int i, int j) {
			return hiddenWeights[i][j];
		}
		
		public double getOutputWeight(int i, int j) {
			return outputWeights[i][j];
		}
		
		public void setHiddenWeight(int i, int j, double weight) {
			hiddenWeights[i][j] = weight;
		}
		
		public void setOutputWeight(int i, int j, double weight) {
			outputWeights[i][j] = weight;
		}
		
		public void setHiddenBias(int i, double weight) {
			hiddenBiases[i] = weight;
		}
		
		public void setOutputBias(int i, double weight) {
			outputBiases[i] = weight;
		}
		
		public void setFitness(double fitness) {
		        this.fitness = fitness;
		}

		public double getFitness() {
		        return fitness;
		}
		
		@Override
		public int nextMove(int[] currentState) {
			double [] hiddenOutput = new double [hiddenDim];
			int outputOutput = 0;
			double aux = 0;
			double aux2 = 0;
			for(int i=0; i < hiddenDim; i++) {
				for(int j=0; j < INPUTDIM; j++) {
					 aux += currentState[j] * hiddenWeights[j][i];
				}
				aux += hiddenBiases[i];	
			hiddenOutput[i]= sigmoid(aux);
			}
			
			
			for(int i=0; i< OUTPUTDIM; i++) {
				for (int j = 0; j < hiddenDim; j++) {
					aux2 += hiddenOutput[j] * outputWeights[j][i];
					
				}
				aux2 += outputBiases[i];
				outputOutput = func(aux2);
			}
			
			return outputOutput;
			}
			
}