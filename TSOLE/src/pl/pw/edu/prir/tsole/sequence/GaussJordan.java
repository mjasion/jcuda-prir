package pl.pw.edu.prir.tsole.sequence;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.io.IOLogic;

public class GaussJordan {
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
				matrixA[i][j] = matrixA[i][j]/wsp;
			}
			matrixB[i][0] = matrixB[i][0]/wsp;
			
			for(int j=0; j<m; j++) {
				if(j==i) continue;
				
				wsp = matrixA[j][i];
				for(int k=0; k<n; k++) {
					matrixA[j][k] = matrixA[j][k] - matrixA[i][k] * wsp;
				}
				
				matrixB[j][0] = matrixB[j][0] - matrixB[i][0] * wsp;
			}
			
		}
		return matrixB;
	}
	
	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
		float[][] matrixA = IOLogic.readMatrix("matrixA");
		float[][] matrixB = IOLogic.readMatrix("matrixB");
		GaussJordan gj = new GaussJordan(matrixA, matrixB);
		IOLogic.printMatrix(gj.run());
	}
}
