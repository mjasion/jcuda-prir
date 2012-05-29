package pl.pw.edu.prir.tsole.sequence;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.io.IOLogic;

public class Gauss implements ISequenceAlgorithm {
	private float[][] matrixA;
	private float[][] matrixB;
//	private float[][] matrixA = {{3, 2, 1} , {2, 3, 1}, {1, 1, 4}};
//	private float[][] matrixB = {{11}, {13}, {12}};
	public Gauss(float[][] A,  float[][] B) {
		this.matrixA = A;
		this.matrixB = B;
	}
	
	public float[][] run() {
		float wsp;
		int m = matrixA.length;
		int n = matrixA[0].length;
		IOLogic.printMatrix(matrixA);
		IOLogic.printMatrix(matrixB);
		for(int i = 0; i<m; i++) {
			wsp = matrixA[i][i];
			for(int j=0; j<n; j++) {
				matrixA[i][j] = matrixA[i][j]/wsp;
			}
			matrixB[i][0] = matrixB[i][0]/wsp;
			for(int j=i+1; j<m; j++) {
				wsp = matrixA[j][i];
				for(int k=0; k<n; k++) {
					matrixA[j][k] = matrixA[j][k] - matrixA[i][k] * wsp;
				}
				matrixB[j][0] = matrixB[j][0] - matrixB[i][0]*wsp;
			}
			IOLogic.printMatrix(matrixA);
			IOLogic.printMatrix(matrixB);
			
		}
		IOLogic.printMatrix(matrixB);
		float[][] result = new float[m][1];
		for(int i=m-1; i>=0; i--) {
//			System.out.println(i);
			float s = matrixB[i][0];
			
			for(int j=n-1; j>i; j--) {
				System.out.println(j + " " + s + " " + matrixA[i][j]);
				s = s - matrixA[i][j] * result[j][0];
			}
			result[i][0] = s;
		}
		
		return result;
	}
	
	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
		float[][] matrixA = IOLogic.readMatrix("matrixA");
		float[][] matrixB = IOLogic.readMatrix("matrixB");
		Gauss gj = new Gauss(matrixA, matrixB);
		IOLogic.printMatrix(gj.run());
	}
}
