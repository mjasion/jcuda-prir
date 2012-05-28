package pl.pw.edu.prir.tsole.howto;

import java.util.ArrayList;
import java.util.List;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

/**
 * 
 * @author marek
 * 
 * 
 *         Macierz przechowywana jest jako wektor + informacja o ilosci kolumn
 * 
 * 
 *         -interesujace funkcje
 * 
 *         cublasSsymm
 * 
 *         cublasStrmm - mnozenie macierzy ?
 * 
 *         cublasStrsm - wyglada na rozwiazanie ukladu rownan
 * 
 *         cublesZgemm
 */
public class Macierze {

	public static void main(String args[]) {

		new Macierze().przechowywanieMacierzy();

		System.out.println("test macierz");

		float macierz[][] = new float[3][4];

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 4; j++) {
				macierz[i][j] = i + j;
				System.out.println(macierz[i][j]);
			}
		}

		System.out.println("macierz length  = " +  macierz.length);
		System.out.println(macierz[0].length);
		
		float wektor[] = macierz[0];

		
		System.out.println(wektor.length);
	}

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

		// wypisanie macierzy A
		System.out.println("A:");
		System.out.println(toString2D(A, matrixSize));

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

		// wypisanie macierzy C
		System.out.println("C:");
		System.out.println(toString2D(C, matrixSize));

		System.out.println("test");
		System.out.println(C[0]);

	}

	private static String toString2D(float[] a, int columns) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < a.length; i++) {
			if (i > 0 && i % columns == 0) {
				sb.append("\n");
			}
			sb.append(String.format("%7.4f ", a[i]));
		}
		return sb.toString();
	}
	
	/**
	 * funckja zamieniajaca macierz na wektor potrzebny do jcudy
	 * 
	 */
	
	public Float[] matrixToVector(Float[][] macierz, int rows, int cols){
		List<Float> jcudaVector = new ArrayList<>();
		
		for(int i =0; i < rows;i++){
			for(int j = 0; j < cols; j++)
				jcudaVector.add(macierz[i][j]);
		}
		
		return (Float[])jcudaVector.toArray();
	}

	/**
	 * funkcje zamieniajca wektor na macierz
	 */
	
	public Float[][] vectorToMAtrix(float[] vector, int cols, int rows){
		
		/**
		 * TODO:
		 * 
		 * 
		 */
		
		return null;
	}
	

}
