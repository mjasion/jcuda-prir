package pl.pw.edu.prir.tsole.sequence;

import java.util.Arrays;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.cuda.TsoleUtils;
import pl.pw.edu.prir.tsole.io.IOLogic;

public class Gauss implements ISequenceAlgorithm {
	private float[][] matrix;
	private static float[][] matrixA = {{0, 2, 1} , {2, 3, 1}, {1, 1, 4}};
	private static float[][] matrixB = {{11}, {13}, {12}};
	public Gauss(float[][] A,  float[][] B) {
		this.matrix = TsoleUtils.joinMatrix(A, B);
	}
	
	public float[][] run() {
		float wsp;
		int m = matrix.length;
		int n = matrix[0].length;
		for(int i = 0; i<m; i++) {
			wsp = matrix[i][i];
			if(wsp == 0) { // szukanie najwiekszego i podmiana
				int nonZeroRowIndex = i;
				for(int j=i+1; j<m; j++) {
					if(matrix[j][i] != 0) {
						nonZeroRowIndex = j;
						break;
					}					
				}
				TsoleUtils.swapRows(matrix, i, nonZeroRowIndex);
				wsp = matrix[i][i];
			}
			
			for(int j=0; j<n; j++) {
				matrix[i][j] = matrix[i][j]/wsp;
			}
			for(int j=i+1; j<m; j++) {
				wsp = matrix[j][i];
				for(int k=0; k<n; k++) {
					matrix[j][k] = matrix[j][k] - matrix[i][k] * wsp;
				}
			}
			IOLogic.printMatrix(matrix);
		}
		float[][] result = new float[m][1];
		for(int i=m-1; i>=0; i--) {
			float s = matrix[i][n-1];
			
			for(int j=n-2; j>i; j--) {
				s = s - matrix[i][j] * result[j][0];
			}
			result[i][0] = s;
		}
		
		return result;
	}
	
	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
//		float[][] matrixA = IOLogic.readMatrix("matrixA");
//		float[][] matrixB = IOLogic.readMatrix("matrixB");
		Gauss gj = new Gauss(matrixA, matrixB);
		IOLogic.printMatrix(gj.run());
	}
}
