package pl.pw.edu.prir.tsole.cuda;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 
 * @author marek
 * 
 * 
 * Utilsy do konwersji macierzy/wektor na takie ,ktore sa zgodne z przechowywaniem w cudzie
 */
public class TsoleUtils {


	/**
	 * funckja zamieniajaca macierz na wektor potrzebny do jcudy
	 * 
	 */

	public static float[] matrixToVector(float[][] macierz, int rows, int cols) {
		List<Float> tempVector = new ArrayList<>();

		/* specjalnie odwrotna kolejnosc  aby bylo tak jak to przechowuje jcuda*/

		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++)
				tempVector.add(macierz[j][i]);
		}

		float[] jcudaVector = new float[tempVector.size()];

		for (int i = 0; i < tempVector.size(); i++) {
			jcudaVector[i] = tempVector.get(i);
		}

		return jcudaVector;
	}

	/**
	 * funkcje zamieniajca wektor na macierz
	 */

	public static float[][] vectorToMAtrix(float[] vector, int rows, int cols) {

		float matrix[][] = new float[rows][cols];
		int i = 0;
		
		//wypelnianie macierzy idac po kolumnach 

		for (int j = 0; j < cols; j++) {
			for (int k = 0; k < rows; k++) {
				matrix[k][j] = vector[i];
				i++;
			}
		}

		return matrix;
	}
	
	
	/**
	 * tworzy wektor akceptowany przez JCude, gdzie ostatnia kolumna sa wartosci z macierzy(wektora) Y // A * X = Y
	 * 
	 * [ 1 4 7 A ]
	 * [ 2 5 8 B ]
	 * [ 3 6 9 C ]
	 *  A, B, C to elementry wektora Y
	 *  
	 * @param matrix
	 * @param rows1
	 * @param cols1
	 * @param matrix2
	 * @return
	 */
	public static float[] makeCombinedVectorFromMatrixes(float[][] matrix, int rows1, int cols1, float[][] matrix2, int rows2, int cols2){
		float[] combinedVector = new float[rows1 * cols1 + rows2 * cols2];
		List<Float> combined = new ArrayList<>();
		
		
		//A
		for (int i = 0; i < rows1; i++) {
			for (int j = 0; j < cols1; j++)
				combined.add(matrix[j][i]);
//				combinedVector[(i* rows1) + j]=matrix[j][i];
		}
		
		//Y- should be a vector ;p
		
		for(int i = 0; i < rows2; i++){
			for(int j =0; j < cols2; j++){
				combined.add(matrix2[j][i]);
			}
		}
		
		int i = 0;
		for(float it: combined)
			combinedVector[i++] = it;
		
		
		return combinedVector;
	}
	
	/**
	 * 
	 * @param vector
	 * 			wektor wynikowy bedaca zlaczeniem ( A | X ) przechowywany w formacie odpowiednim dla cudy
	 * @return
	 * 		float[] results  ostatnia kolumne macierzy pobrana z wektora
	 */
	public static float[] getResultsFromResultVector(float[] vector, int rows, int cols){
		int lenght = rows*cols;
		float[] result = new float[rows];
		
		for(int i =0; i < rows; i++)
			result[i] = vector[lenght - rows + i];
		
		return result;
	}

	public static float det(float[][] mat) {
		float result = 0;

		if (mat.length == 1) {
			result = mat[0][0];
			return result;
		}

		if (mat.length == 2) {
			result = mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
			return result;
		}

		for (int i = 0; i < mat[0].length; i++) {
			float temp[][] = new float[mat.length - 1][mat[0].length - 1];

			for (int j = 1; j < mat.length; j++) {
				System.arraycopy(mat[j], 0, temp[j - 1], 0, i);
				System.arraycopy(mat[j], i + 1, temp[j - 1], i, mat[0].length - i - 1);
			}

			result += mat[0][i] * Math.pow(-1, i) * det(temp);
		}

		return result;

	}
	
	public static int getMaxRowIndex(float[][] matrix, int index, int rowBeg, int rowEnd) {
		int nonZeroRowIndex = rowBeg;
		float max = matrix[index][index];
		
		for(int j=rowBeg; j<rowEnd; j++) {
			if(Math.abs(matrix[j][index]) > max ) {
				nonZeroRowIndex = j;
				max = Math.abs(matrix[j][index]);
			}					
		}
		
		return nonZeroRowIndex;
	}
	
	public static void swapRows(float[][] matrix, int i, int nonZeroRowIndex) {
		int length =  matrix[i].length;
		float[] row = Arrays.copyOf(matrix[i], length);
		matrix[i] = Arrays.copyOf(matrix[nonZeroRowIndex], length);
		matrix[nonZeroRowIndex] = Arrays.copyOf(row, length);
	}
	

	public static float[][] joinMatrix(float[][] A, float[][] B) {
		float[][] matrix = new float[A.length][A[0].length + 1];
		int n = A[0].length; // pozycja na ktora wrzucac B
		int m = A.length;
		
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				matrix[i][j] = A[i][j];
			}
			matrix[i][n] = B[i][0];
		}

		return matrix;
	}
	
	public static void printMatrix(float[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;
		System.out.println(m + "x" + n);
		for(int i=0; i<m; i++) {
			System.out.println(Arrays.toString(matrix[i]));
		}
		
	}
	
	public static void printMatrix(double[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;
		System.out.println(m + "x" + n);
		for(int i=0; i<m; i++) {
			System.out.println(Arrays.toString(matrix[i]));
		}
		
	}
	
	
	public static void printMatrix(float[] matrix) {
		int m = matrix.length;
		System.out.println(m + "x" + 1);
		for(int i=0; i<m; i++) {
			System.out.println("[" + matrix[i] + "]");
		}
		
	}
	
}
