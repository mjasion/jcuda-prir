package pl.pw.edu.prir.tsole;

import jcuda.LogLevel;
import jcuda.jcublas.JCublas;

import org.apache.log4j.PropertyConfigurator;

import pl.pw.edu.prir.tsole.cuda.TsoleUtils;
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
 * 	1. dorobic sprawdzanie czy wymiary macierzy sa odpowiednie
 * 	2. dodatkowy argument w wywolania pozwalajacy uzyskac logowanie dzialania jcublas np. -l
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

		TsoleUtils.printMatrix(matrix);
		TsoleUtils.printMatrix(matrix2);


		// inicjalizacja jcublas
		JCublas.cublasInit();
		JCublas.setLogLevel(LogLevel.LOG_TRACE);
		
		/**
		 * TODO:
		 * wywolywanie cudy np. CudaGaussJordan...
		 */
		
		//finalizacja
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
