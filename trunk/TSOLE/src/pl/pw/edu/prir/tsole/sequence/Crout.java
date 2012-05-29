package pl.pw.edu.prir.tsole.sequence;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.io.IOLogic;

public class Crout implements ISequenceAlgorithm{
	private float[][] matrixA;
	private float[][] matrixB;
	public Crout(float[][] A,  float[][] B) {
		this.matrixA = A;
		this.matrixB = B;
	}
	
	public float[][] run() {


		return matrixB;
	}
	
	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
		float[][] matrixA = IOLogic.readMatrix("matrixA");
		float[][] matrixB = IOLogic.readMatrix("matrixB");
		Crout gj = new Crout(matrixA, matrixB);
		IOLogic.printMatrix(gj.run());
	}
}
