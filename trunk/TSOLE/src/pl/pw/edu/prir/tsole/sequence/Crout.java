package pl.pw.edu.prir.tsole.sequence;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.cuda.TsoleUtils;
import pl.pw.edu.prir.tsole.io.IOLogic;

public class Crout implements ISequenceAlgorithm{
	private float[][] matrix;
	public Crout(float[][] A,  float[][] B) {
		matrix = TsoleUtils.joinMatrix(A, B);
	}
	
	public float[][] run() {


		return matrix;
	}
	
	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
		float[][] matrixA = IOLogic.readMatrix("matrixA");
		float[][] matrixB = IOLogic.readMatrix("matrixB");
		Crout gj = new Crout(matrixA, matrixB);
		TsoleUtils.printMatrix(gj.run());
	}
}
