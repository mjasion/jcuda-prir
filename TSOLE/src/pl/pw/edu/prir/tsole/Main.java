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


/**
 * 
 * TODO:
 * 	-
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
		// matrix A
		Pointer pa = new Pointer();
		float cudaVector_A[] = TsoleCuda.matrixToVector(matrix, rows, cols);
		// vector B
		Pointer pb = new Pointer();
		float cudaVector_B[] = TsoleCuda.matrixToVector(matrix2, 1, rows);
		
		System.out.println("print vector B");
		IOLogic.printMatrix(cudaVector_B);
//		float cudaVector_B[] = TsoleCuda.matrixToVector(matrix2, rowsB, colsB);
	
		// matrix C
		Pointer pc = new Pointer();
		float cudaVector_C[] = new float[rows];

		// inicjalizacja jcublas
		JCublas.cublasInit();
		JCublas.setLogLevel(LogLevel.LOG_TRACE);
		
		// alokacja pamieci
		if(JCublas.cublasAlloc(rows * cols, Sizeof.FLOAT, pa) != 0)
			System.err.println("bład alokacji wskaznika pa");
		
		if(JCublas.cublasAlloc(rows , Sizeof.FLOAT, pb) != 0)
			System.err.println("bład alokacji wskaznika pb");
		
		if(JCublas.cublasAlloc(rows , Sizeof.FLOAT, pc) != 0)
			System.err.println("bład alokacji wskaznika pc");
		
		// przerzucenie danych do gpu
		JCublas.cublasSetMatrix(rows, cols, Sizeof.FLOAT, Pointer.to(cudaVector_A), rows, pa, cols);
		JCublas.cublasSetVector(rows, Sizeof.FLOAT, Pointer.to(cudaVector_B), 1, pb, 1);
		
		JCublas.printVector(rows, pb);
		JCublas.cublasSaxpy(1, 2, pb, 2, pc, 2);
		JCublas.printVector(rows, pc);
		
		// wywolanie funkcji ,ktora beda obliczac w cudzie
		/* 1. Gauss-Jordan */
		
			/* a) dla kazdego wiersza
			 * 			wsp = matrix[i][i]
			 */
		

		JCublas.printMatrix(rows, pa, cols);
//		JCublas.printVector(1, pb);
//			float wsp;
//			for(int i =0; i < rows; i++){
//				wsp = matrix[i][i];
//				
//				//podziel macierz A
//				for(int j=0; j < cols; j++){
//					
//				}
//				//podziel wektor B
//				
//				for(int k=0; k < rows; k++){
//					if(k==i) continue;	
//					
//					
//					
//					
//				}
//				
//			}

		
		
		/*-----------------*/
		
		//pobranie wynikow
		JCublas.cublasGetMatrix(rows, cols, Sizeof.FLOAT, pb, rows, Pointer.to(cudaVector_C), cols);
		JCublas.cublasGetVector(rows, Sizeof.FLOAT, pb, 1, Pointer.to(cudaVector_C), 1);
		
		System.out.println("iologic print");
		IOLogic.printMatrix(cudaVector_C);
		
		
		
		//zwolnienie pamieci
		JCublas.cublasFree(pa);
		JCublas.cublasFree(pb);
		JCublas.cublasFree(pc);
		
		//
		JCublas.cublasShutdown();

		// Float matrixAfter[][] = Macierze.vectorToMAtrix(cudaVector_A,
		// matrix.length, matrix[0].length);

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
