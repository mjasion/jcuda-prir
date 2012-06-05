package pl.pw.edu.prir.tsole.sequence;

import org.apache.log4j.Logger;
import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.cuda.TsoleUtils;
import pl.pw.edu.prir.tsole.io.IOLogic;

public class GaussJordan implements ISequenceAlgorithm {
	private static final Logger log = Logger.getLogger(GaussJordan.class);

	private float[][] matrix;
	private static float[][] matrixA = { { 0, 2, 1 }, { 2, 0, 1 }, { 1, 1, 4 } };
	private static float[][] matrixB = { { 11 }, { 13 }, { 12 } };

	public GaussJordan(float[][] A, float[][] B) {
		matrix = TsoleUtils.joinMatrix(A, B);
	}

	public float[][] run() {
		float wsp;
		int m = matrix.length;
		int n = matrix[0].length;

		long start = System.nanoTime();
		long end;
		
		for (int i = 0; i < m; i++) {
			wsp = matrix[i][i];

			if (wsp == 0) {
				int nonZeroRowIndex = TsoleUtils.getMaxRowIndex(matrix, i, 0, m);
				TsoleUtils.swapRows(matrix, i, nonZeroRowIndex);
				wsp = matrix[i][i];
			}

			for (int j = 0; j < n; j++) {
				matrix[i][j] = matrix[i][j] / wsp;
			}
			for (int j = 0; j < m; j++) {
				if (j == i)
					continue;

				wsp = matrix[j][i];
				for (int k = 0; k < n; k++) {
					matrix[j][k] = matrix[j][k] - matrix[i][k] * wsp;
				}
			}

		}

		float[][] result = new float[m][1];
		for (int i = 0; i < m; i++) {
			result[i][0] = matrix[i][n - 1];
		}

		end = System.nanoTime();
		System.out.println("\n[Gauss-Jordan] czas obliczeń : "+ (end-start)+ " ns. [" + ((end-start)/1000000000.00)+" s]");
		
		return result;
	}

	public static void main(String... args) {
		PropertyConfigurator.configure("log4j.properties");
		float[][] A = IOLogic.readMatrix("matrixA");
		float[][] B = IOLogic.readMatrix("matrixB");
//		GaussJordan gj = new GaussJordan(A, B);
		 GaussJordan gj = new GaussJordan(matrixA, matrixB);
		TsoleUtils.printMatrix(gj.run());
	}
}
