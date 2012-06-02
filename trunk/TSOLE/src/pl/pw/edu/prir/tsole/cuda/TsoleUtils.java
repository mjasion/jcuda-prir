package pl.pw.edu.prir.tsole.cuda;

import static jcuda.jcublas.JCublas.cublasAlloc;
import static jcuda.jcublas.JCublas.cublasFree;
import static jcuda.jcublas.JCublas.cublasGetMatrix;
import static jcuda.jcublas.JCublas.cublasGetVector;
import static jcuda.jcublas.JCublas.cublasIsamax;
import static jcuda.jcublas.JCublas.cublasSetMatrix;
import static jcuda.jcublas.JCublas.cublasSetVector;
import static jcuda.jcublas.JCublas.cublasSgemm;
import static jcuda.jcublas.JCublas.cublasSgemv;
import static jcuda.jcublas.JCublas.cublasSger;
import static jcuda.jcublas.JCublas.cublasSscal;
import static jcuda.jcublas.JCublas.cublasSswap;
import static jcuda.jcublas.JCublas.cublasStrmv;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;

import java.util.ArrayList;
import java.util.List;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

/**
 * 
 * @author marek
 * 
 *         -interesujace funkcje
 * 
 *         cublasSsymm
 * 
 *         cublasStrmm - 
 * 
 *         cublasStrsm -
 * 
 *         cublesZgemm
 *         
 *         Klasa pomocnicza sluzaca do konwersji macierzy/wektora do i z gpu.
 */
public class TsoleUtils {

	/**
	 * 
	 * @param size
	 * 		wymiar macierzy(1)
	 * @param A 
	 * 		macierz
	 */
	public static void invertMatrix(int size, float A[]) {
		Pointer pa = new Pointer();
		// alokacja
		cublasAlloc(size * size, Sizeof.FLOAT, pa);
		// kopiowanie na gpu
		cublasSetMatrix(size, size, Sizeof.FLOAT, Pointer.to(A), size, pa, size);

		invertMatrix(size, pa);

		// odebranie z gpu
		cublasGetMatrix(size, size, Sizeof.FLOAT, pa, size, Pointer.to(A), size);
		// porzadki
		cublasFree(pa);
	}

	public static void invertMatrix(int n, Pointer pa) {
		// LU
		int pivots[] = factorizeLU(n, pa);

		doInversion(n, pa, pivots);
	}
	
	/*
	 * 
	 */

	/**
	 * 
	 * @param size
	 * 		wym macierzy
	 * @param pa
	 * 		macierz(wskaznik)
	 * @return 
	 * 		os
	 */
	private static int[] factorizeLU(int size, Pointer pa) {
		int[] pivots = new int[size];

		for (int i = 0; i < size; i++) {
			pivots[i] = i;
		}

		float factor[] = { 0.0f };
		Pointer pf = Pointer.to(factor);
		
		for (int i = 0; i < size - 1; i++) {
			//po diag.
			Pointer offset = at(pa, i * size + i);
				
			int pivot = i - 1 + cublasIsamax(size - i, offset, 1);
			if (pivot != i) {
				pivots[i] = pivot;
				cublasSswap(size, at(pa, pivot), size, at(pa, i), size);
			}
			//pobr z pamieci
			cublasGetVector(1, Sizeof.FLOAT, offset, 1, pf, 1);
			// x = alpha*x
			cublasSscal(size - i - 1, 1 / factor[0], at(offset, 1), 1);
			
			cublasSger(size - i - 1, size - i - 1, -1.0f, at(offset, 1), 1, at(offset, size), size, at(offset, size + 1), size);
		}
		
		return pivots;
	}
	
	/**
	 * 
	 * 
	 * @param n
	 * @param pa
	 * @param pivots
	 */

	private static void doInversion(int n, Pointer pa, int[] pivots) {
		
		inverseUpperTriangle(n, pa);

		// inv(A)*L = inv(U)
		Pointer dWork = new Pointer();
		cublasAlloc(n - 1, Sizeof.FLOAT, dWork);

		for (int i = n - 1; i > 0; i--) {
			Pointer offset = at(pa, ((i - 1) * n + i));
			cudaMemcpy(dWork, offset, (n - 1) * Sizeof.FLOAT, cudaMemcpyDeviceToDevice);
			cublasSscal(n - i, 0, offset, 1);
			cublasSgemv('n', n, n - i, -1.0f, at(pa, i * n), n, dWork, 1, 1.0f, at(pa, ((i - 1) * n)), 1);
		}

		cublasFree(dWork);

		// Pivot back to original order
		for (int i = n - 1; i >= 0; i--) {
			if (i != pivots[i]) {
				cublasSswap(n, at(pa, i * n), 1, at(pa, pivots[i] * n), 1);
			}
		}

	}

	private static void inverseUpperTriangle(int size, Pointer pa) {
		float[] factor = { 0.0f };
		Pointer pFactor = Pointer.to(factor);
		for (int i = 0; i < size; i++) {
			Pointer offset = at(pa, i * size);
			cublasGetVector(1, Sizeof.FLOAT, at(offset, i), 1, pFactor, 1);
			factor[0] = 1 / factor[0];
			cublasSetVector(1, Sizeof.FLOAT, pFactor, 1, at(offset, i), 1);
			cublasStrmv('u', 'n', 'n', i, pa, size, offset, 1);
			// x = alpha*x
			cublasSscal(i, -factor[0], offset, 1);
		}
	}

	private static Pointer at(Pointer p, int floatOffset) {
		return p.withByteOffset(floatOffset * Sizeof.FLOAT);
	}

	/**
	 * mnozenie macierzy wynikowa C zostaje nadpisana
	 * 
	 * @param size
	 * @param A
	 * @param B
	 * @param C
	 */

	public static void multiply(int size, float A[], float B[], float C[]) {
		Pointer pa = new Pointer();
		Pointer pb = new Pointer();
		Pointer pc = new Pointer();

		cublasAlloc(size * size, Sizeof.FLOAT, pa);
		cublasAlloc(size * size, Sizeof.FLOAT, pb);
		cublasAlloc(size * size, Sizeof.FLOAT, pc);
		cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(A), 1, pa, 1);
		cublasSetVector(size * size, Sizeof.FLOAT, Pointer.to(B), 1, pb, 1);

		cublasSgemm('n', 'n', size, size, size, 1, pa, size, pb, size, 0, pc, size);

		cublasGetVector(size * size, Sizeof.FLOAT, pc, 1, Pointer.to(C), 1);
		cublasFree(pa);
		cublasFree(pb);
		cublasFree(pc);
	}

	/*
	 * testing
	 */
	public static void main(String args[]) {

		new TsoleUtils().przechowywanieMacierzy();

		System.out.println("test macierz");

		float macierz[][] = new float[3][4];

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				macierz[i][j] = i + j;
				System.out.println(macierz[i][j]);
			}
		}

		System.out.println("macierz length  = " + macierz.length);
		System.out.println(macierz[0].length);

		float wektor[] = macierz[0];

		System.out.println(wektor.length);
	}

	/**
	 * testowanko
	 */
	public void przechowywanieMacierzy() {
		System.out.println("dodawanie macierzy");

		int matrixSize = 5;
		// macierz A
		float A[] = new float[matrixSize * matrixSize];
		Pointer pa = new Pointer();

		// macierz C
		float C[] = new float[matrixSize * matrixSize];
		Pointer pc = new Pointer();

		for (int i = 0; i < matrixSize * matrixSize; i++) {
			A[i] = i;
			C[i] = 0;
		}

		// inicjalizacja
		JCublas.cublasInit();

		// alokacja pamieci
		int result = JCublas.cublasAlloc(matrixSize * matrixSize, Sizeof.FLOAT, pa);
		System.out.println("result alokacji = " + result);

		// przekopiowanie macierzy do gpu
		result = 13;
		result = JCublas.cublasSetMatrix(matrixSize, matrixSize, Sizeof.FLOAT, Pointer.to(A), matrixSize, pa, matrixSize);
		// result = JCublas.cublasSetVector(matrixSize * matrixSize,
		// Sizeof.FLOAT, Pointer.to(A), 1, pa, 1);
		System.out.println("result przenoszenia = " + result);

		// wypisanie macierzy bedacej w gpu pod pa
		System.out.println("macierz w gpu");
		// JCublas.printMatrix(matrixSize, pa, matrixSize);

		// pobranie macierzy z gpu do macierzy C
		JCublas.cublasGetMatrix(matrixSize, matrixSize, Sizeof.FLOAT, pa, matrixSize, Pointer.to(C), matrixSize);

		// JCublas.cublasGetVector(matrixSize* matrixSize, Sizeof.FLOAT,
		// Pointer.to(pa), 1, C,1);

		JCublas.cublasFree(pa);

		// finito
		JCublas.cublasShutdown();

		System.out.println("test");
		System.out.println(C[0]);

	}

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

	

}
