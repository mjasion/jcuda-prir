package pl.pw.edu.prir.tsole;

import jcuda.LogLevel;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.cuda.TsoleCuda;
import pl.pw.edu.prir.tsole.io.IOLogic;

/**
 * Parametry VM: -Djava.library.path=${workspace_loc:TSOLE/JCuda-0.4.1/lib}
 * 
 * @author Marcin, marek
 * 
 */

public class Main {

	private static String pathToMatrixA = null;
	private static String pathToMatrixB = null;
	private static String pathToOutputFile = null;
	private static String methodSelected = null;

	public static void main(String[] args) {
		PropertyConfigurator.configure("log4j.properties");

		if (Manual.parseArgs(args) == 1)
			return;

		Manual.printWelcome();

		float[][] matrix = IOLogic.readMatrix(pathToMatrixA);
		float[][] matrix2 = IOLogic.readMatrix(pathToMatrixB);
		int rows = matrix.length;
		int cols = matrix[0].length;

		IOLogic.printMatrix(matrix);
		IOLogic.printMatrix(matrix2);

		/* do jcudy */
		//macierz(wektor) A
		float cudaVector_A[] = TsoleCuda.matrixToVector(matrix, rows, cols);
		// odwrotnosc A
		float invA[] = cudaVector_A.clone();
		// wektor y(B)
		float cudaVector_B[] = TsoleCuda.matrixToVector(matrix2, 1, rows);
		// macierz(wektor) X
		float cudaVector_C[] = new float[rows];

		for (int i = 0; i < rows; i++)
			cudaVector_C[i] = 0;

		// inicjalizacja jcublas
		JCublas.cublasInit();
		// JCublas.setLogLevel(LogLevel.LOG_TRACE);


		TsoleCuda.invertMatrix(rows, invA);
		float jednostkowa[] = new float[rows * rows];
		TsoleCuda.multiply(rows, cudaVector_A, invA, jednostkowa);


	
//		System.out.println("macierz jednostkowa");
//		IOLogic.printMatrix(TsoleCuda.vectorToMAtrix(jednostkowa, rows, cols));
		System.out.println("\nmacierz A");
		IOLogic.printMatrix(TsoleCuda.vectorToMAtrix(cudaVector_A, rows, cols));
		System.out.println("\nmacierz inv A");
		IOLogic.printMatrix(TsoleCuda.vectorToMAtrix(invA, rows, cols));
		System.out.println("\nwektor X");
		TsoleCuda.multiply(rows, invA, cudaVector_B, cudaVector_C);
		IOLogic.printMatrix(cudaVector_C);
		
		JCublas.cublasShutdown();
	}

	public static String getPathToMatrixA() {
		return pathToMatrixA;
	}

	public static String getPathToMatrixB() {
		return pathToMatrixB;
	}

	public static String getMethodSelected() {
		return methodSelected;
	}

	public static void setPathToMatrixA(String pathToMatrixA) {
		Main.pathToMatrixA = pathToMatrixA;
	}

	public static void setPathToMatrixB(String pathToMatrixB) {
		Main.pathToMatrixB = pathToMatrixB;
	}

	public static void setMethodSelected(String methodSelected) {
		Main.methodSelected = methodSelected;
	}

	public static String getPathToOutputFile() {
		return pathToOutputFile;
	}

	public static void setPathToOutputFile(String pathToOutputFile) {
		Main.pathToOutputFile = pathToOutputFile;
	}

}
