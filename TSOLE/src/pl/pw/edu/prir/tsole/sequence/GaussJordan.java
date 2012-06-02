package pl.pw.edu.prir.tsole.sequence;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.io.IOLogic;

public class GaussJordan implements ISequenceAlgorithm{
	private float[][] matrixA;
	private float[][] matrixB;
	public GaussJordan(float[][] A,  float[][] B) {
		this.matrixA = A;
		this.matrixB = B;
	}
	
	public float[][] run() {
		float wsp;
		int m = matrixA.length;
		int n = matrixA[0].length;
	
		for(int i = 0; i<m; i++) {
			wsp = matrixA[i][i];
			for(int j=0; j<n; j++) {
				matrixA[i][j] = matrixA[i][j]/wsp;	//2.9
			}
			matrixB[i][0] = matrixB[i][0]/wsp;		// 2.9 - na macierzy B ;]
			
			for(int j=0; j<m; j++) {			// 2.9 - 2.24 :D - cala reszta
				if(j==i) continue;				// jesli ta sama linia co wyzej, to pomin
				
				wsp = matrixA[j][i];			// wspolczynnik rozwiazywanego rownania(np. a21)
				for(int k=0; k<n; k++) {		
					matrixA[j][k] = matrixA[j][k] - matrixA[i][k] * wsp;	// 2.10 i 2.11 mnozenie a nastepnie odejmowanie 
				}
				
				matrixB[j][0] = matrixB[j][0] - matrixB[i][0] * wsp;		// 2.10 i 2.11 - dla macierzy B
			}
			
		}
		return matrixB;
	}
	
	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
		float[][] matrixA = IOLogic.readMatrix("matrixA");
		float[][] matrixB = IOLogic.readMatrix("matrixB");
//		float[][] matrixA = IOLogic.readMatrix("matrix_simple_a");
//		float[][] matrixB = IOLogic.readMatrix("matrix_simple_b");

		GaussJordan gj = new GaussJordan(matrixA, matrixB);
		IOLogic.printMatrix(gj.run());
	}
}
